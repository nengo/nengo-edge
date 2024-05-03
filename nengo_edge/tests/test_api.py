# pylint: disable=missing-docstring

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Union

import pytest
import requests
from dotenv import load_dotenv
from gql.transport.exceptions import TransportQueryError
from graphql.error.graphql_error import GraphQLError
from pytest import MonkeyPatch

from nengo_edge import api

load_dotenv()


class MockResponse:
    def __init__(self, status_code: int, json_data: Dict[str, Any]):
        self.status_code = status_code
        self.data = json_data

    def json(self) -> Dict[str, Any]:
        return self.data

    @property
    def ok(self) -> bool:
        return self.status_code < 400


@pytest.fixture(scope="module", name="test_token")
def fixture_test_token() -> Union[str, None]:
    # Local Testing: Set a token (must be updated every 24 hours)
    token = os.environ.get("NENGO_EDGE_API_TEST_TOKEN")
    if token is not None:
        return token
    # CI: Get a Client Credentials Token
    secret = os.environ.get("AUTH_CLIENT_SECRET")
    auth_domain = os.environ.get(api.AUTH_DOMAIN_KEY)
    client_id = os.environ.get(api.CLIENT_ID_KEY)
    audience = os.environ.get(api.AUDIENCE_KEY)

    if secret is None or auth_domain is None or client_id is None or audience is None:
        pytest.xfail(
            "One or more of AUTH_CLIENT_SECRET, {api.AUTH_DOMAIN_KEY}, "
            "{api.CLIENT_ID_KEY} or {api.AUDIENCE_KEY} not set"
        )

    print("Fetching new auth token")
    token = requests.post(
        url=f"{auth_domain}oauth/token",
        headers={"content-type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": secret,
            "audience": audience,
        },
        timeout=30,
    ).json()["access_token"]
    return token


def test_create_client_defaults(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token", "https://example.com")
    assert client.base_url == "https://example.com"

    monkeypatch.delenv(api.API_ROOT_KEY)

    client = api.NengoEdgeClient("token")
    assert client.base_url == "https://edge.nengo.ai/gql"

    client = api.NengoEdgeClient("token", "https://example.com")
    assert client.base_url == "https://example.com"


def test_has_unexpired_token() -> None:
    token_client = api.NengoEdgeTokenClient("client_id")
    assert token_client.has_unexpired_token() is False

    token_client.token_expiry = datetime.now() + timedelta(seconds=30)
    assert token_client.has_unexpired_token() is False

    token_client.token = "fake_token"
    assert token_client.has_unexpired_token() is True

    token_client.token_expiry = datetime.now() - timedelta(seconds=1)
    assert token_client.has_unexpired_token() is False


def test_get_token(monkeypatch: MonkeyPatch) -> None:
    def mock_request_token(self: api.NengoEdgeTokenClient) -> str:
        self.token = "new_token"
        self.token_expiry = datetime.now() + timedelta(seconds=3)
        return self.token

    monkeypatch.setattr(api.NengoEdgeTokenClient, "request_token", mock_request_token)

    token_client = api.NengoEdgeTokenClient("client_id")
    token_client.token = "original_token"
    token_client.token_expiry = datetime.now() + timedelta(seconds=3)
    token = token_client.get_token()
    assert token == "original_token"

    # Expired
    token_client.token_expiry = datetime.now() - timedelta(seconds=3)
    token = token_client.get_token()
    assert token == "new_token"

    # Force skip cache
    token_client.token = "another_token"
    token = token_client.get_token(True)
    assert token == "new_token"


def test_get_projects_succeeds(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query: {
            "projects": [
                {"_id": "123", "name": "Project 1 Name"},
                {"_id": "234", "name": "Project 2 Name"},
            ]
        },
    )
    result = client.get_projects()
    assert result == [
        {"_id": "123", "name": "Project 1 Name"},
        {"_id": "234", "name": "Project 2 Name"},
    ]


def test_get_datasets_succeeds(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query, variables: {
            "baseData": [
                {
                    "datasets": [
                        {"_id": "123", "name": "Dataset 1 Name"},
                        {"_id": "234", "name": "Dataset 2 Name"},
                    ]
                }
            ]
        },
    )
    result = client.get_datasets("KWS")
    assert result == [
        {"_id": "123", "name": "Dataset 1 Name"},
        {"_id": "234", "name": "Dataset 2 Name"},
    ]


def test_get_networks_succeeds(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query, variables: {
            "networks": [
                {"_id": "123", "name": "Network 1 Name"},
            ]
        },
    )
    result = client.get_networks("KWS")
    assert result == [
        {"_id": "123", "name": "Network 1 Name"},
    ]


def test_add_run(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query, variables: {
            "addRun": [
                {"_id": "123"},
            ]
        },
    )
    result = client.add_run("123", "KWS")
    assert result == [
        {"_id": "123"},
    ]


def test_get_results(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query, variables: {
            "run": {
                "status": "QUEUED",
                "results": [
                    {"category": "CLASSIFICATION", "name": "accuracy", "value": 90}
                ],
            }
        },
    )
    result = client.get_results("123")
    assert result == {
        "status": "QUEUED",
        "results": [{"category": "CLASSIFICATION", "name": "accuracy", "value": 90}],
    }


def test_update_run(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query, variables: {
            "updateRun": [
                {"_id": "123"},
            ]
        },
    )
    result = client.update_run("123", {"param": "value"})
    assert result == [
        {"_id": "123"},
    ]


def test_start_optimize(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")
    monkeypatch.setattr(
        client,
        "_execute_gql",
        lambda query, variables: {
            "startOptimize": [
                {"_id": "123"},
            ]
        },
    )
    result = client.start_optimize("123")
    assert result == [
        {"_id": "123"},
    ]


@pytest.mark.parametrize(
    "hardware,dataset,network",
    [
        ("gpu", "datasetId", "networkId"),
        ("cpu", None, None),
        ("tsp", None, "networkId"),
    ],
)
def test_start_training_run_creates_correct_input(
    monkeypatch: MonkeyPatch, hardware: str, dataset: str, network: str
) -> None:

    def mock_add_run(projectId: str, model_type: str) -> Dict[str, Any]:
        return {"_id": "fakeRunId"}

    def mock_update_run(run_id: str, run_input: Dict[str, Any]) -> Dict[str, Any]:
        # Assert run_input is as expected for test
        assert run_input["hardwareId"] == hardware

        assert run_input.get("datasetId") == dataset
        assert run_input.get("networkId") == network
        assert run_input["hyperparams"]["steps"] == 1000
        return {}

    def mock_start_optimize(run_id: str) -> Dict[str, Any]:
        return {}

    client = api.NengoEdgeClient("token")

    monkeypatch.setattr(client, "add_run", mock_add_run)
    monkeypatch.setattr(client, "update_run", mock_update_run)
    monkeypatch.setattr(client, "start_optimize", mock_start_optimize)

    run_id = client.start_training_run(
        "projectId", "KWS", {"steps": 1000}, hardware, dataset, network
    )
    assert run_id == "fakeRunId"


def test_start_training_run_succeeds(test_token: str) -> None:
    if test_token is None:
        pytest.skip("Test token not available")
    client = api.NengoEdgeClient(test_token)
    model_type = "KWS"
    project_id = client.get_projects()[0]["_id"]
    dataset_id = "65e8bfa226c1057c10a6f658"
    network_id = client.get_networks(model_type)[0]["_id"]

    run_id = client.start_training_run(
        project_id, model_type, {"steps": 1}, "cpu", dataset_id, network_id
    )
    assert run_id is not None
    results = client.get_results(run_id)
    assert results["status"] == "QUEUED"
    assert len(results["results"]) == 0


def test_get_device_code_succeeds(monkeypatch: MonkeyPatch) -> None:
    def mock_post(url: str, data: Dict[str, Any], timeout: int) -> MockResponse:
        return MockResponse(
            200,
            {
                "verification_uri_complete": "http://example.com/code",
                "device_code": "code",
                "interval": 1,
                "expires_in": 3,
            },
        )

    token_client = api.NengoEdgeTokenClient()

    monkeypatch.setattr(token_client.session, "post", mock_post)

    device_code = token_client.get_device_code()
    assert device_code["verification_uri_complete"] == "http://example.com/code"


def test_request_token_succeeds(monkeypatch: MonkeyPatch) -> None:
    token_client = api.NengoEdgeTokenClient()

    def mock_get_device_code() -> Dict[str, Any]:
        return {
            "verification_uri_complete": "http://example.com/code",
            "device_code": "code",
            "interval": 1,
            "expires_in": 3,
        }

    monkeypatch.setattr(
        token_client,
        "get_device_code",
        mock_get_device_code,
    )

    def mock_get_token_from_device_code(device_code: Dict[str, Any]) -> Dict[str, Any]:
        return {"access_token": "mock_token", "expires_in": 30}

    monkeypatch.setattr(
        token_client,
        "get_token_from_device_code",
        mock_get_token_from_device_code,
    )

    token = token_client.request_token()
    requested_time = datetime.now()

    assert token_client.token == token
    assert token == "mock_token"
    assert token_client.token_expiry is not None
    assert token_client.token_expiry <= requested_time + timedelta(seconds=30)


def test_get_device_code_fails(monkeypatch: MonkeyPatch) -> None:
    def mock_post(url: str, data: Dict[str, Any], timeout: int) -> MockResponse:
        return MockResponse(
            403,
            {
                "error": "unauthorized_client",
                "error_description": "Unauthorized or unknown client",
            },
        )

    token_client = api.NengoEdgeTokenClient("invalid client id")

    monkeypatch.setattr(token_client.session, "post", mock_post)

    with pytest.raises(api.AuthenticationError):
        token_client.get_device_code()


def test_get_token_from_device_code_succeeds(monkeypatch: MonkeyPatch) -> None:
    first_post_call = True

    def mock_post(url: str, data: Dict[str, Any], timeout: int) -> MockResponse:
        nonlocal first_post_call
        if first_post_call:
            first_post_call = False
            return MockResponse(403, {"error": "authorization_pending"})
        return MockResponse(200, {"access_token": "mock_token", "expires_in": 60})

    token_client = api.NengoEdgeTokenClient()

    monkeypatch.setattr(token_client.session, "post", mock_post)

    device_code = {
        "verification_uri_complete": "http://example.com/code",
        "device_code": "code",
        "interval": 1,
        "expires_in": 3,
    }

    response = token_client.get_token_from_device_code(device_code)
    assert response["access_token"] == "mock_token"
    assert response["expires_in"] == 60


def test_get_token_from_device_code_fails(monkeypatch: MonkeyPatch) -> None:
    def mock_post(url: str, data: Dict[str, Any], timeout: int) -> MockResponse:
        return MockResponse(
            403,
            {
                "error": "invalid_grant",
                "error_description": "Invalid or expired device code.",
            },
        )

    token_client = api.NengoEdgeTokenClient()

    monkeypatch.setattr(token_client.session, "post", mock_post)

    device_code = {
        "verification_uri_complete": "http://example.com/code",
        "device_code": "code",
        "interval": 1,
        "expires_in": 3,
    }

    with pytest.raises(api.AuthenticationError):
        token_client.get_token_from_device_code(device_code)


def test_get_token_from_device_code_times_out(monkeypatch: MonkeyPatch) -> None:
    def mock_post(url: str, data: Dict[str, Any], timeout: int) -> MockResponse:
        return MockResponse(403, {"error": "authorization_pending"})

    token_client = api.NengoEdgeTokenClient()

    monkeypatch.setattr(token_client.session, "post", mock_post)

    device_code = {
        "verification_uri_complete": "http://example.com/code",
        "device_code": "code",
        "interval": 1,
        "expires_in": 2,
    }

    with pytest.raises(
        api.AuthenticationError, match="Timed out waiting for confirmation."
    ):
        token_client.get_token_from_device_code(device_code)


def test_create_token_client_defaults(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv(api.AUTH_DOMAIN_KEY)
    monkeypatch.delenv(api.CLIENT_ID_KEY)
    monkeypatch.delenv(api.AUDIENCE_KEY)

    with pytest.raises(ValueError):
        api.NengoEdgeTokenClient()

    client = api.NengoEdgeTokenClient("client_id")
    assert client.client_id == "client_id"
    assert client.auth_domain == "https://auth.nengo.ai/"
    assert client.audience == "https://edge.nengo.ai"
    assert client.session.headers["content-type"] == "application/x-www-form-urlencoded"

    client = api.NengoEdgeTokenClient(
        "client_id", "https://auth.example.com/", "https://example.com/"
    )
    assert client.client_id == "client_id"
    assert client.auth_domain == "https://auth.example.com/"
    assert client.audience == "https://example.com/"
    assert client.session.headers["content-type"] == "application/x-www-form-urlencoded"


def test_execute_gql_errors(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")

    def mock_execute_gql_error(
        query: str, variable_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise GraphQLError(
            "Cannot query field 'project' on type 'Query'. Did you mean 'projects'?"
        )

    monkeypatch.setattr(client.client, "execute", mock_execute_gql_error)
    invalid_query = """
            query GetAllProjects {
                project {
                _id
                }
            }
            """
    with pytest.raises(api.RequestError, match="Invalid GraphQL Query"):
        client._execute_gql(invalid_query)


def test_execute_gql_expired_token(monkeypatch: MonkeyPatch) -> None:
    client = api.NengoEdgeClient("token")

    def mock_execute_gql_error(
        query: str, variable_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise TransportQueryError("", errors=[{"message": "Token is expired"}])

    monkeypatch.setattr(client.client, "execute", mock_execute_gql_error)
    valid_query = """
            query GetAllProjects {
                projects {
                _id
                name
                }
            }
            """
    with pytest.raises(api.AuthenticationError, match="Token is expired"):
        client._execute_gql(valid_query)
