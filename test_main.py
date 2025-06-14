import unittest
from unittest.mock import MagicMock, patch
import json
import requests
from fastapi.testclient import TestClient
from main import app, get_schema_search
from services.sql_generation_service import SQLGenerationService

class TestGenerateSQLLogic(unittest.TestCase):
    
    def setUp(self):
        self.schema_text = """
        Table: users
        Columns:
        - id (NUMBER, NOT NULL)
        - name (VARCHAR2, NOT NULL)
        - email (VARCHAR2, NULL)
        - created_at (DATE, NULL)
        """
        self.nl_query = "Find all users created in the last week"
        self.expected_sql = "SELECT * FROM users WHERE created_at >= SYSDATE - 7"

    @patch('services.sql_generation_service.MISTRAL_API_KEY', 'fake_api_key')
    @patch('requests.post')
    def test_generate_sql_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': self.expected_sql}}]
        }
        mock_post.return_value = mock_response

        sql_query, error = SQLGenerationService.generate_sql(self.schema_text, self.nl_query)
        self.assertEqual(sql_query, self.expected_sql)
        self.assertIsNone(error)

    @patch('services.sql_generation_service.MISTRAL_API_KEY', '')
    def test_generate_sql_no_api_key(self):
        sql_query, error = SQLGenerationService.generate_sql(self.schema_text, self.nl_query)
        self.assertIsNone(sql_query)
        self.assertIn("Mistral API key not found", error)

    @patch('services.sql_generation_service.MISTRAL_API_KEY', 'fake_api_key')
    @patch('requests.post')
    def test_generate_sql_http_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError("API rate limit exceeded")
        sql_query, error = SQLGenerationService.generate_sql(self.schema_text, self.nl_query)
        self.assertIsNone(sql_query)
        self.assertIn("Mistral API error", error)


    @patch('services.sql_generation_service.MISTRAL_API_KEY', 'fake_api_key')
    @patch('requests.post')
    def test_generate_sql_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        sql_query, error = SQLGenerationService.generate_sql(self.schema_text, self.nl_query)
        self.assertIsNone(sql_query)
        self.assertIn("Connection refused", error)

    @patch('services.sql_generation_service.MISTRAL_API_KEY', 'fake_api_key')
    @patch('requests.post')
    def test_generate_sql_different_model(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': self.expected_sql}}]
        }
        mock_post.return_value = mock_response

        sql_query, error = SQLGenerationService.generate_sql(self.schema_text, self.nl_query, model="mistral-medium")
        self.assertEqual(sql_query, self.expected_sql)
        self.assertIsNone(error)

    @patch('services.sql_generation_service.MISTRAL_API_KEY', 'fake_api_key')
    @patch('requests.post')
    def test_generate_sql_cleans_response(self, mock_post):
        wrapped_sql = "```sql\n" + self.expected_sql + "\n```"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': wrapped_sql}}]
        }
        mock_post.return_value = mock_response

        sql_query, error = SQLGenerationService.generate_sql(self.schema_text, self.nl_query)
        self.assertEqual(sql_query, self.expected_sql)
        self.assertIsNone(error)


class TestGenerateSQLAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('services.sql_generation_service.SQLGenerationService.generate_sql')
    def test_generate_sql_endpoint(self, mock_generate_sql):
        mock_generate_sql.return_value = ("SELECT * FROM users", None)

        # Fake schema search object
        class FakeSchemaSearch:
            def search(self, query):
                return [{
                    "table_name": "users",
                    "similarity_score": 0.95,
                    "description": "User table"
                }]

        # ✅ Proper override
        app.dependency_overrides[get_schema_search] = lambda: FakeSchemaSearch()

        schema_data = {
            "users": [
                {"column_name": "id", "data_type": "NUMBER", "nullable": "N"},
                {"column_name": "name", "data_type": "VARCHAR2", "nullable": "N"},
                {"column_name": "created_at", "data_type": "DATE", "nullable": "Y"},
            ]
        }

        with patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=json.dumps(schema_data)):
            response = self.client.post(
                "/generate-sql",
                json={"query": "Find all users", "model": "mistral-large-latest"}
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sql_query"], "SELECT * FROM users")
        self.assertEqual(data["nl_query"], "Find all users")
        self.assertIsInstance(data["execution_time"], float)
        self.assertEqual(len(data["relevant_tables"]), 1)

        # ✅ Cleanup
        app.dependency_overrides = {}

if __name__ == "__main__":
    unittest.main()
