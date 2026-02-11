"""
Intent parsing for natural language queries.

This module provides LLM-based parsing of natural language queries
into structured QueryIntent objects.
"""

import re
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from deepgraph.core.schema import GraphSchema
from deepgraph.planning.dsl import (
    QueryIntent,
    QueryType,
    QueryPlan,
    QUERY_PATTERNS,
    STRATEGY_MAPPING,
    SYNTHESIS_PROMPT_MAPPING
)


class IntentParser:
    """LLM-based intent parser.

    This class uses an LLM to parse natural language queries into
    structured QueryIntent objects that can be executed.

    Example:
        parser = IntentParser(
            openai_client=client,
            schema=ARXIV_SCHEMA
        )

        intent = parser.parse("What papers has Geoffrey Hinton written?")
        # Returns: QueryIntent(
        #     query_type=QueryType.AUTHOR_PAPERS,
        #     entities=["Geoffrey Hinton"],
        #     target_label="Paper",
        #     ...
        # )
    """

    PARSING_PROMPT = """You are a query intent parser for a graph RAG system over academic papers.

Given a natural language query, extract:
1. Query type (similarity, author_papers, collaborator_papers, connection_path, comparison, summary, etc.)
2. Entity names mentioned (authors, paper titles, concepts)
3. Target entity type (Paper, Author, Concept, Method)
4. Any constraints (date ranges, categories, etc.)

Graph Schema:
- Nodes: Paper, Author, Concept, Method
- Edges: AUTHORED_BY, CO_AUTHORED, ABOUT_CONCEPT, USES_METHOD, CITES

Respond with JSON only:
{{
  "query_type": "author_papers",
  "entities": ["Geoffrey Hinton"],
  "target_label": "Paper",
  "constraints": {{}},
  "reasoning_depth": 2,
  "top_k": 10,
  "source_entity": null,
  "target_entity": null,
  "confidence": 0.95
}}

Query Types:
- similarity: Find papers about a topic (vector search)
- author_papers: Find papers by specific authors
- collaborator_papers: Find papers by collaborators
- connection_path: Find how entities are connected
- comparison: Compare papers, authors, or methods
- summary: Summarize information about an entity
- author_network: Analyze author collaboration networks

Examples:

Query: "What papers discuss transformers in NLP?"
Response: {{"query_type": "similarity", "entities": ["transformers", "NLP"], "target_label": "Paper", "constraints": {{}}, "reasoning_depth": 1, "top_k": 10, "confidence": 0.95}}

Query: "What has Geoffrey Hinton written?"
Response: {{"query_type": "author_papers", "entities": ["Geoffrey Hinton"], "target_label": "Paper", "constraints": {{}}, "reasoning_depth": 2, "top_k": 10, "confidence": 0.98}}

Query: "Who has Yann LeCun collaborated with?"
Response: {{"query_type": "collaborator_papers", "entities": ["Yann LeCun"], "target_label": "Author", "constraints": {{}}, "reasoning_depth": 3, "top_k": 10, "confidence": 0.97}}

Query: "How are Geoffrey Hinton and Yoshua Bengio connected?"
Response: {{"query_type": "connection_path", "entities": [], "source_entity": "Geoffrey Hinton", "target_entity": "Yoshua Bengio", "target_label": "Author", "constraints": {{}}, "reasoning_depth": 6, "top_k": 1, "confidence": 0.96}}

Query: "Compare BERT and GPT"
Response: {{"query_type": "comparison", "entities": ["BERT", "GPT"], "comparison_entities": ["BERT", "GPT"], "target_label": "Paper", "constraints": {{}}, "reasoning_depth": 2, "top_k": 5, "confidence": 0.94}}

Now parse this query:"""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        schema: Optional[GraphSchema] = None,
        model: str = "gpt-4-turbo-preview"
    ):
        """Initialize intent parser.

        Args:
            openai_client: Pre-configured OpenAI client
            api_key: OpenAI API key (if client not provided)
            schema: Graph schema for domain context
            model: LLM model to use for parsing
        """
        self._client = openai_client or OpenAI(api_key=api_key)
        self._schema = schema
        self._model = model

    def parse(self, query: str) -> QueryIntent:
        """Parse natural language query into structured intent.

        Args:
            query: Natural language query

        Returns:
            QueryIntent object

        Example:
            intent = parser.parse("What papers has Geoffrey Hinton written?")
        """
        # Try pattern-based parsing first (faster)
        pattern_intent = self._pattern_based_parse(query)
        if pattern_intent and pattern_intent.confidence > 0.8:
            return pattern_intent

        # Fall back to LLM parsing for complex queries
        return self._llm_based_parse(query)

    def _pattern_based_parse(self, query: str) -> Optional[QueryIntent]:
        """Fast pattern-based parsing using regex.

        Args:
            query: Natural language query

        Returns:
            QueryIntent if pattern matched, None otherwise
        """
        query_lower = query.lower()

        # Check patterns in order of specificity (more specific first)
        priority_order = [
            QueryType.AUTHOR_PAPERS,
            QueryType.COLLABORATOR_PAPERS,
            QueryType.CONNECTION_PATH,
            QueryType.COMPARISON,
            QueryType.AUTHOR_NETWORK,
            QueryType.SIMILARITY,  # Most general, check last
        ]

        for query_type in priority_order:
            if query_type not in QUERY_PATTERNS:
                continue

            patterns = QUERY_PATTERNS[query_type]
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Extract entities (simplified)
                    entities = self._extract_entities_simple(query)

                    return QueryIntent(
                        query_type=query_type,
                        entities=entities,
                        target_label="Paper",  # Default
                        top_k=10,
                        reasoning_depth=2,
                        confidence=0.7  # Lower confidence for pattern matching
                    )

        return None

    def _llm_based_parse(self, query: str) -> QueryIntent:
        """LLM-based parsing for complex queries.

        Args:
            query: Natural language query

        Returns:
            QueryIntent object
        """
        # Build prompt
        prompt = self.PARSING_PROMPT + f"\n\nQuery: \"{query}\"\nResponse:"

        # Call LLM
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent parsing
            max_tokens=500
        )

        # Parse JSON response
        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            intent_data = json.loads(response_text)

            # Convert to QueryIntent
            intent = QueryIntent(
                query_type=QueryType(intent_data["query_type"]),
                entities=intent_data.get("entities", []),
                target_label=intent_data.get("target_label"),
                relationship_types=intent_data.get("relationship_types", []),
                constraints=intent_data.get("constraints", {}),
                reasoning_depth=intent_data.get("reasoning_depth", 2),
                top_k=intent_data.get("top_k", 10),
                source_entity=intent_data.get("source_entity"),
                target_entity=intent_data.get("target_entity"),
                comparison_entities=intent_data.get("comparison_entities", []),
                confidence=intent_data.get("confidence", 0.9)
            )

            return intent

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to similarity search on parse error
            return QueryIntent(
                query_type=QueryType.SIMILARITY,
                entities=[],
                target_label="Paper",
                top_k=10,
                confidence=0.5
            )

    def _extract_entities_simple(self, query: str) -> list:
        """Simple entity extraction (look for quoted text or proper nouns).

        Args:
            query: Query text

        Returns:
            List of entity strings
        """
        entities = []

        # Extract quoted text
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        # Extract capitalized phrases (potential names)
        # This is simplified - a real implementation would use NER
        words = query.split()
        current_phrase = []
        for word in words:
            if word and word[0].isupper() and word.isalpha():
                current_phrase.append(word)
            else:
                if current_phrase:
                    entities.append(" ".join(current_phrase))
                    current_phrase = []

        if current_phrase:
            entities.append(" ".join(current_phrase))

        return list(set(entities))  # Remove duplicates

    def create_plan(self, query: str) -> QueryPlan:
        """Parse query and create execution plan.

        Args:
            query: Natural language query

        Returns:
            QueryPlan ready for execution

        Example:
            plan = parser.create_plan("What papers discuss transformers?")
            # Can be executed directly
        """
        intent = self.parse(query)

        # Map intent to execution strategies
        reasoning_strategy = STRATEGY_MAPPING.get(intent.query_type)
        synthesis_template = SYNTHESIS_PROMPT_MAPPING.get(
            intent.query_type,
            "default"
        )

        # Build retrieval parameters
        retrieval_params = {
            "top_k": intent.top_k
        }
        if intent.constraints:
            retrieval_params["filters"] = intent.constraints

        # Build reasoning parameters
        reasoning_params = {
            "max_depth": intent.reasoning_depth
        }

        # Add strategy-specific parameters
        if intent.query_type == QueryType.CONNECTION_PATH:
            reasoning_params["target_id"] = intent.target_entity
            reasoning_params["from_label"] = intent.target_label or "Author"
            reasoning_params["to_label"] = intent.target_label or "Author"

        # Build synthesis config
        synthesis_config = {
            "temperature": 0.3,
            "max_tokens": 1000
        }

        plan = QueryPlan(
            intent=intent,
            retrieval_strategy="vector",
            retrieval_params=retrieval_params,
            reasoning_strategy=reasoning_strategy,
            reasoning_params=reasoning_params,
            synthesis_config=synthesis_config,
            synthesis_prompt_template=synthesis_template,
            use_planning=True
        )

        return plan

    def __repr__(self) -> str:
        """String representation."""
        return f"IntentParser(model={self._model})"
