import json
import asyncio
import time
import re
from typing import Dict, List, Any, Tuple
from enum import Enum

from pydantic import BaseModel, Field
from rich.console import Console
from rich import print as rprint

from jsonschema import validate, ValidationError
from groq import AsyncGroq, RateLimitError


from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding

class ProcessingStrategy(str, Enum):
    SINGLE_PASS = "single_pass"
    HIERARCHICAL = "hierarchical"

class ComplexityScore(BaseModel):
    total_fields: int = 0
    nesting_levels: int = 0
    field_types: dict = {}
    extraction_chunks: List = []
    complexity_score: float = 0

class ConversionResult(BaseModel):
    data: Dict[str, Any]
    confidence_justifications: Dict[str, str] = Field(default_factory=dict)
    flagged_fields: List[str] = Field(default_factory=list)
    processing_stats: Dict[str, Any]
    strategy_used: ProcessingStrategy


class SchemaAnalyzer:
    def analyze(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        analysis = ComplexityScore()
        
        def traverse_schema(schema_obj, path="", level=0):
            analysis.nesting_levels = max(analysis.nesting_levels, level)
            
            if isinstance(schema_obj, dict):
                if 'type' in schema_obj:
                    field_type = schema_obj['type']
                    analysis.field_types[field_type] = analysis.field_types.get(field_type, 0) + 1
                    analysis.total_fields += 1
                    
                if 'properties' in schema_obj:
                    properties = schema_obj['properties']
                    chunk_fields = {}
                    for prop_name, prop_schema in properties.items():
                        full_path = f"{path}.{prop_name}" if path else prop_name
                        chunk_fields[prop_name] = prop_schema
                        traverse_schema(prop_schema, full_path, level + 1)
                    
                    if chunk_fields:
                        analysis.extraction_chunks.append({
                            'path': path,
                            'fields': chunk_fields,
                            'level': level
                        })
                elif 'items' in schema_obj:
                    traverse_schema(schema_obj['items'], path, level + 1)
        
        traverse_schema(schema)
        analysis.complexity_score = self._calculate_complexity(analysis)
        return analysis
    
    def _calculate_complexity(self, analysis: ComplexityScore) -> int:
        """Calculate complexity score for resource allocation"""
        score = 0
        score += analysis.total_fields * 1
        score += analysis.nesting_levels * 10
        score += analysis.field_types.get('object', 0) * 5
        score += analysis.field_types.get('array', 0) * 3
        return score


class RAGHandler:
    def __init__(self):
        Settings.embed_model = FastEmbedEmbedding(cache_dir='./models')
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        self.index = None
        console.log("RAG Handler initialized.")

    def ingest_document(self, text: str):
        console.log(f"Ingesting document of {len(text)} characters into vector store...")
        documents = [Document(text=text)]
        self.index = VectorStoreIndex.from_documents(documents)
        console.log("Document ingestion and indexing complete.")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if not self.index:
            raise ValueError("Document not ingested. Please call 'ingest_document' first.")
        
        console.log(f"Retrieving context for query: '[cyan]{query}[/cyan]'")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(query)
        
        context = "\n---\n".join([node.get_content() for node in retrieved_nodes])
        console.log(f"Retrieved {len(retrieved_nodes)} context snippets.")
        return context


class ExtractionPipeline:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.query_gen_model = "llama-3.1-8b-instant"
        
    async def _generate_rag_query_from_schema(self, schema: Dict[str, Any]) -> str:
        console.log("Generating dynamic RAG query from schema...")
        
        properties = schema.get("properties", {})
        field_details = []
        for name, details in properties.items():
            description = details.get("description", "")
            field_details.append(f"- {name}: {description}".strip())

        field_details = '\n'.join(field_details)
        
        prompt = f"""
        You are a search query formulation expert. Based on the following JSON schema fields that need to be populated, generate a single, concise, 
        and effective natural language question that would be used to find the relevant information in a large document. DONT RETURN ANYTHING EXCEPT THE QUESTION

        Schema Fields to fill:
        {field_details}

        Generated Question:
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.query_gen_model, 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            query = response.choices[0].message.content.strip().replace('"', '')
            return query
        except Exception as e:
            console.log(f"[yellow]Warning: Could not generate dynamic query, falling back to simple query. Error: {e}[/yellow]")
            return f"Information about {', '.join(properties.keys())}"

    async def _extract_with_llm(self, prompt: str) -> Tuple[Dict, Dict]:
        max_retries = 3
        backoff = 2
        timeout = 60  
        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.6,
                        max_tokens=8000,
                        response_format={"type": "json_object"}
                    ),
                    timeout=timeout
                )
                break
            except Exception as e:
                    if (
                        (hasattr(e, "code") and getattr(e, "code", None) == 429)
                        or (hasattr(e, "message") and "rate limit" in str(e).lower())
                        or (isinstance(e, RateLimitError) if "RateLimitError" in globals() else False)
                    ):
                        wait_match = re.search(r"try again in ([0-9]+)m([0-9.]+)s", str(e))
                        if wait_match:
                            minutes = int(wait_match.group(1))
                            seconds = float(wait_match.group(2))
                            wait_time = minutes * 60 + seconds
                        else:
                            wait_time = backoff ** attempt
                        if attempt == max_retries:
                            raise
                        console.log(f"[yellow]Rate limit hit, retrying in {wait_time:.1f} seconds (attempt {attempt}/{max_retries})[/yellow]")
                        await asyncio.sleep(wait_time)
                    else:
                        if attempt == max_retries:
                            raise TimeoutError(f"Groq API call timed out after {timeout} seconds.")
                        console.log(f"[yellow]Timeout, retrying in {backoff ** attempt} seconds (attempt {attempt}/{max_retries})[/yellow]")
                        await asyncio.sleep(backoff ** attempt)
        else:
            raise RuntimeError("Failed to get response from Groq after retries.")
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("data", {}), result.get("justification", {})
        except (json.JSONDecodeError, AttributeError):
            console.log("[yellow]Warning: Could not decode JSON from LLM response. Trying regex fallback.[/yellow]")
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return result.get("data", result), result.get("justification", {})
                except json.JSONDecodeError:
                    return {}, {}
            return {}, {}

    async def extract_single_pass(
        self, rag_handler: RAGHandler, schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        query = await self._generate_rag_query_from_schema(schema)
        context = rag_handler.retrieve_context(query)
        
        prompt = f"""
        Based on the given CONTEXT, extract information and format it according to the provided JSON schema.
        Your response MUST be a single JSON object with two keys: "data" and "justification".
        - "data": The JSON object that strictly follows the schema.
        - "justification": A JSON object where each key is a field path (e.g., "user.address.city") and the value is a brief explanation of why you extracted that value.
        
        If you cannot find information for a field, use the default value for the schema field type. Be precise.
        
        SCHEMA:
        {json.dumps(schema, indent=2)}
        
        CONTEXT:
        ---
        {context}
        ---
        
        """
        return await self._extract_with_llm(prompt)
    
    async def extract_hierarchical(
        self, rag_handler: RAGHandler, schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        final_data = {}
        final_justification = {}
        
        top_level_props = {k: v for k, v in schema.get("properties", {}).items() if v.get("type") != "object" and v.get("type") != "array"}
        top_level_schema = {"type": "object", "properties": top_level_props}
        console.log(f"Pass for Top Level Fields : '[bold cyan]{top_level_schema}[/bold cyan]'")
        if top_level_props:
            data, justification = await self.extract_single_pass(rag_handler, top_level_schema)
            final_data.update(data)
            final_justification.update(justification)

        for field_name, field_schema in schema.get("properties", {}).items():
            if field_schema.get("type") == "object" or field_schema.get("type") == "array":
                console.log(f"Hierarchical pass for nested field: '[bold magenta]{field_name}[/bold magenta]'")
                nested_schema = {"type": "object", "properties": {field_name: field_schema}}
                data, justification = await self.extract_single_pass(rag_handler, nested_schema)
                if field_name in data:
                    final_data[field_name] = data[field_name]
                    final_justification.update(justification)

        return final_data, final_justification

    
def sanitize_data_for_schema(data: Dict, schema: Dict) -> Dict:
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return data

    # Look at the properties defined in the current schema level
    properties = schema.get("properties", {})
    for key, prop_schema in properties.items():
        expected_type = prop_schema.get("type")
        
        # If the key exists in our data
        if key in data:
            if data[key] is None and expected_type == "array":
                console.log(f"[yellow]Sanitizing '{key}': Converted None to empty array [].[/yellow]")
                data[key] = []
            elif data[key] is None and expected_type == "object":
                console.log(f"[yellow]Sanitizing '{key}': Converted None to empty object {{}}.[/yellow]")
                data[key] = {}
            elif data[key] is None and expected_type == "string":
                console.log(f"[yellow]Sanitizing '{key}': Converted None to empty string ''.[/yellow]")
                data[key] = ''
            elif isinstance(data[key], dict) and expected_type == "object":
                data[key] = sanitize_data_for_schema(data[key], prop_schema)
            elif isinstance(data[key], list) and expected_type == "array":
                item_schema = prop_schema.get("items", {})
                data[key] = [sanitize_data_for_schema(item, item_schema) for item in data[key]]

    return data

class JSONValidator:
    def validate_extraction(
        self, data: Dict, schema: Dict, justifications: Dict, confidence_keyword: str = "guess"
    ) -> Tuple[bool, List[str]]:
        try:
            validate(instance=data, schema=schema)
            schema_valid = True
        except ValidationError as e:
            console.log(f"[red]Schema validation failed: {e} [/red]")
            schema_valid = False
        
        flagged_fields = [
            # field for field, justification in justifications.items()
            # if confidence_keyword in justification.lower()
        ]
        return schema_valid, flagged_fields


class Unstructured2Structured:
    def __init__(self, groq_api_key: str, model: str):
        self.schema_analyzer = SchemaAnalyzer()
        self.rag_handler = RAGHandler()
        self.extraction_pipeline = ExtractionPipeline(api_key=groq_api_key, model=model)
        self.validator = JSONValidator()
    
    def select_strategy(self, complexity: ComplexityScore) -> ProcessingStrategy:
        if complexity.complexity_score < 50: return ProcessingStrategy.SINGLE_PASS
        return ProcessingStrategy.HIERARCHICAL
    
    async def convert(self, text: str, schema: Dict) -> ConversionResult:
        start_time = time.time()
        
        complexity: ComplexityScore = self.schema_analyzer.analyze(schema)
        strategy = self.select_strategy(complexity)

        self.rag_handler.ingest_document(text)

        console.rule(f"[bold green]Executing Strategy: {strategy.value}[/bold green]")
        if strategy == ProcessingStrategy.SINGLE_PASS:
            data, justifications = await self.extraction_pipeline.extract_single_pass(self.rag_handler, schema)
        elif strategy == ProcessingStrategy.HIERARCHICAL:
            data, justifications = await self.extraction_pipeline.extract_hierarchical(self.rag_handler, schema)
    
        sanitized_data = sanitize_data_for_schema(data, schema)
        
        schema_valid, flagged_fields = self.validator.validate_extraction(sanitized_data, schema, justifications)
        
        processing_stats = {
            "processing_time_seconds": round(time.time() - start_time, 2),
            "strategy_used": strategy.value,
            "complexity_score": round(complexity.complexity_score, 2),
            "schema_valid": schema_valid,
            "total_fields_extracted": len(data),
            "fields_flagged": len(flagged_fields)
        }
        
        return ConversionResult(
            data=sanitized_data,
            confidence_justifications=justifications,
            flagged_fields=flagged_fields,
            processing_stats=processing_stats,
            strategy_used=strategy
        )

console = Console()
