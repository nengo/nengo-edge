import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from gql import Client, gql
from gql.transport.httpx import HTTPXAsyncTransport

AUDIENCE_KEY = "NENGO_EDGE_API_AUDIENCE"
AUTH_DOMAIN_KEY = "NENGO_EDGE_API_AUTH_DOMAIN"
CLIENT_ID_KEY = "NENGO_EDGE_API_CLIENT_ID"
API_ROOT_KEY = "NENGO_EDGE_API_ROOT"


class NengoEdgeTokenClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        auth_domain: Optional[str] = None,
        audience: Optional[str] = None,
    ):
        if client_id is None:
            client_id = os.environ.get(CLIENT_ID_KEY)
            if client_id is None:
                raise ValueError(
                    "client_id not be set as a parameter or using"
                    f"{CLIENT_ID_KEY} environment variable"
                )
        self.client_id = client_id

        if auth_domain is None:
            auth_domain = os.environ.get(AUTH_DOMAIN_KEY)
            if auth_domain is None:
                auth_domain = "https://auth.nengo.ai/"
        self.auth_domain = auth_domain

        if audience is None:
            audience = os.environ.get(AUDIENCE_KEY)
            if audience is None:
                audience = "https://edge.nengo.ai"
        self.audience = audience

        self.token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

    def has_unexpired_token(self) -> bool:
        return (
            self.token_expiry is not None
            and self.token_expiry < datetime.now()
            and self.token is not None
        )

    async def get_token(self, skip_cache: bool = False) -> str:
        if skip_cache or not self.has_unexpired_token():
            return await self.request_token()
        assert self.token is not None
        return self.token

    async def request_token(self) -> str:
        device_code = await self.get_device_code()
        requested_time = datetime.now()
        token_response = await self.get_token_from_device_code(device_code)
        token = token_response["access_token"]
        delta = timedelta(seconds=token_response["expires_in"])
        self.token = token
        self.token_expiry = requested_time + delta
        return token

    async def get_device_code(self) -> Dict[str, Any]:
        scope = "read write"

        headers = {"content-type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient() as session:
            device_code = (
                await session.post(
                    url=f"{self.auth_domain}oauth/device/code",
                    headers=headers,
                    data={
                        "scope": scope,
                        "client_id": self.client_id,
                        "audience": self.audience,
                    },
                )
            ).json()

        return device_code

    async def get_token_from_device_code(
        self, device_code: Dict[str, Any]
    ) -> Dict[str, Any]:
        headers = {"content-type": "application/x-www-form-urlencoded"}

        print(f"Go to {device_code['verification_uri_complete']} to confirm access")

        max_time = device_code["expires_in"]  # May want to convert to clock time
        interval = device_code["interval"]

        await asyncio.sleep(interval)
        elapsed_time = interval

        async with httpx.AsyncClient() as session:
            while elapsed_time < max_time:
                token_response = await session.post(
                    url=f"{self.auth_domain}oauth/token",
                    headers=headers,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": device_code["device_code"],
                        "client_id": self.client_id,
                    },
                )
                token_response_json = token_response.json()

                if token_response.status_code == 200:
                    return token_response_json

                if (
                    token_response.status_code == 403
                    and token_response_json["error"] == "authorization_pending"
                ):
                    elapsed_time += interval
                    print("Waiting for confirmation in browser")
                    await asyncio.sleep(interval)
                else:
                    # Some other error
                    print(token_response_json)
                    raise NotImplementedError

        return token_response_json


class NengoEdgeClient:
    def __init__(self, token: str, base_url: Optional[str] = None):
        if base_url is None:
            base_url = os.environ.get(API_ROOT_KEY)
        if base_url is None:
            base_url = "https://edge.nengo.ai/gql"
        self.base_url = base_url

        self._create_client(token)

    def _create_client(self, token: str) -> None:
        transport = HTTPXAsyncTransport(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {token}",
            },
        )
        self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def set_token(self, new_token: str) -> None:
        self._create_client(new_token)

    async def _execute_gql(
        self, gql_string: str, variables: Optional[Dict] = None
    ) -> Dict:
        query_or_mutation = gql(gql_string)
        # Wrong Permissions: TransportQueryError
        return await self.client.execute_async(
            query_or_mutation, variable_values=variables
        )

    async def get_projects(self) -> List[Dict[str, Any]]:
        response = await self._execute_gql(
            """
            query GetAllProjects {
                projects {
                _id
                name           
                }
            }
            """
        )
        return response["projects"]

    async def add_run(self, project_id: str, model_type: str) -> Dict[str, Any]:
        add_run_mutation = """
        mutation AddRun($id: ObjectId!, $modelType: ModelType!) {
            addRun(project: $id, modelType: $modelType) {
            _id
            }
        }
        """

        return await self._execute_gql(
            add_run_mutation, {"id": project_id, "modelType": model_type}
        )

    async def update_run(
        self, run_id: str, run_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        update_run_mutation = """
        mutation UpdateRun($id: ObjectId!, $runInput: RunInput!) {
            updateRun(run: $id, updates: $runInput) {
            _id
            }
        }
        """
        return await self._execute_gql(
            update_run_mutation,
            variables={"id": run_id, "runInput": run_input},
        )

    async def start_optimize_run(self, run_id: str) -> dict:
        start_optimize_mutation = """
            mutation StartOptimize($id: ObjectId!) {
                startOptimize(run: $id) {
                _id
                status
                }
            }
        """
        return await self._execute_gql(
            start_optimize_mutation, variables={"id": run_id}
        )

    async def get_results(self, run_id: str) -> Dict:
        get_results_query = """
            query GetResults($id: ObjectId!) {
                run(_id: $id) {
                    status
                    results {
                        category
                        name
                        value
                    }
                }
            }
        """
        response = await self._execute_gql(get_results_query, variables={"id": run_id})
        return response["run"]

    # Higher level functions
    async def start_training_run(
        self, project_id: str, model_type: str, hyperparams: dict[str, Any]
    ) -> str:
        add_response = await self.add_run(project_id, model_type)
        run_id = add_response["addRun"]["_id"]
        run_input = {"hyperparams": hyperparams}
        await self.update_run(run_id, run_input)
        await self.start_optimize_run(run_id)
        return run_id
