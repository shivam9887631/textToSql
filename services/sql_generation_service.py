import requests
from typing import Tuple, Optional
from config import MISTRAL_API_KEY, MISTRAL_MODELS

class SQLGenerationService:
    @staticmethod
    def generate_sql(schema_text: str, nl_query: str, model: str = "mistral-large-latest") -> Tuple[Optional[str], Optional[str]]:
        if not MISTRAL_API_KEY:
            return None, "Mistral API key not found. Please check your .env file."

        if model not in MISTRAL_MODELS:
            return None, f"Invalid model. Choose from: {', '.join(MISTRAL_MODELS)}"

        # Construct prompt
        prompt = f"""
        You are an SQL expert. Convert the following natural language query into a detailed SQL query.

        Database Schema (including table details):
        {schema_text}

        User Query: {nl_query}

        Follow these guidelines:
        1. Use appropriate JOINs based on table relationships
        2. Add WHERE filters based on query
        3. Use aliases for tables if needed
        4. Add aggregate functions if relevant
        5. Include ORDER BY / GROUP BY / HAVING if required
        6. Add comments if necessary

        Return ONLY the SQL query without explanations or markdown formatting.
        """

        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }
            )

            response.raise_for_status()  # Raise exception if status != 200

            try:
                result = response.json()
            except ValueError:
                return None, "Error: Received invalid JSON from Mistral API."

            sql_query = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Remove markdown ```sql blocks if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]

            return sql_query.strip(), None

        except requests.exceptions.HTTPError as e:
            return None, f"Mistral API error: {str(e)}"
        except requests.exceptions.RequestException as e:
            return None, f"Connection error: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
