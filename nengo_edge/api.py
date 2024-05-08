"""Classes for interacting with the Nengo Edge Server via an API."""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from gql import Client, gql
from gql.transport.exceptions import TransportQueryError
from gql.transport.requests import RequestsHTTPTransport
from graphql.error.graphql_error import GraphQLError

AUDIENCE_KEY = "NENGO_EDGE_API_AUDIENCE"
AUTH_DOMAIN_KEY = "NENGO_EDGE_API_AUTH_DOMAIN"
CLIENT_ID_KEY = "NENGO_EDGE_API_CLIENT_ID"
API_ROOT_KEY = "NENGO_EDGE_API_ROOT"


class NengoEdgeAPIError(Exception):
    """Generic errors relating to NengoEdgeAPI."""


class AuthenticationError(NengoEdgeAPIError):
    """An error because of a problem authenticating."""


class RequestError(NengoEdgeAPIError):
    """An error because of a problem with the request.."""


class NotFoundError(NengoEdgeAPIError):
    """An error because the requested item does not exist."""


class NengoEdgeTokenClient:
    """Class to retrieve and manage tokens for the Nengo Edge API."""

    REQUESTS_TIMEOUT = 7

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

        self.session = requests.Session()
        self.session.headers.update(
            {"content-type": "application/x-www-form-urlencoded"}
        )

    def has_unexpired_token(self) -> bool:
        """
        Check the cache for a token that is not expired.

        Returns
        -------
        bool
            Returns true if the token exists and is not expired.
        """
        return (
            self.token_expiry is not None
            and self.token_expiry > datetime.now()
            and self.token is not None
        )

    def get_token(self, skip_cache: bool = False) -> str:
        """
        Retrieve a token from the cache or the authorization server.

        This requires user interaction if there's not a valid token.

        Parameters
        ----------
        skip_cache : bool
            If true, always requests a new token.

        Returns
        -------
        str
            An access token.
        """
        if skip_cache or not self.has_unexpired_token():
            return self.request_token()
        assert self.token is not None
        return self.token

    def request_token(self) -> str:
        """
        Retrieve a token from the authorization server.

        This requires user interaction.

        Returns
        -------
        str
            An access token.
        """
        device_code = self.get_device_code()
        requested_time = datetime.now()
        token_response = self.get_token_from_device_code(device_code)
        token = token_response["access_token"]
        delta = timedelta(seconds=token_response["expires_in"])
        self.token = token
        self.token_expiry = requested_time + delta
        return token

    def get_device_code(self) -> Dict[str, Any]:
        """
        Request a device code from the authorization server.

        See: https://datatracker.ietf.org/doc/html/rfc8628

        Returns
        -------
        dict
            The device code response per RFC 8628.
        """
        scope = "read write"
        response = self.session.post(
            url=f"{self.auth_domain}oauth/device/code",
            data={
                "scope": scope,
                "client_id": self.client_id,
                "audience": self.audience,
            },
            timeout=self.REQUESTS_TIMEOUT,
        )

        response_obj = response.json()
        if response.ok:
            return response_obj
        else:
            raise AuthenticationError(response_obj)

    def get_token_from_device_code(self, device_code: Dict[str, Any]) -> Dict[str, Any]:
        """
        Requests a token using a device code from the authorization server.

        This requires user interaction - they must navigate to the URL
        and click on a button in the browser. This may be on a separate device.

        See: https://datatracker.ietf.org/doc/html/rfc8628

        Parameters
        ----------
        device_code : dict
            The device code response from `get_device_code`.

        Returns
        -------
        dict
            The OAuth 2.0 token response.
        """
        print(f"Go to {device_code['verification_uri_complete']} to confirm access")

        max_time = device_code["expires_in"]  # May want to convert to clock time
        interval = device_code["interval"]

        time.sleep(interval)
        elapsed_time = interval

        while elapsed_time < max_time:
            token_response = self.session.post(
                url=f"{self.auth_domain}oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code["device_code"],
                    "client_id": self.client_id,
                },
                timeout=self.REQUESTS_TIMEOUT,
            )
            token_response_json = token_response.json()

            if token_response.status_code == 200:
                return token_response_json

            if (
                token_response.status_code == 403
                and token_response_json.get("error") == "authorization_pending"
            ):
                elapsed_time += interval
                print("Waiting for confirmation in browser")
                time.sleep(interval)
            else:
                # Some other error
                raise AuthenticationError(token_response_json)

        raise AuthenticationError("Timed out waiting for confirmation.")


class NengoEdgeClient:
    """Class to make calls to the Nengo Edge API."""

    def __init__(
        self,
        token: str,
        base_url: Optional[str] = None,
        custom_headers: Optional[Dict[str, Any]] = None,
    ):
        if base_url is None:
            base_url = os.environ.get(API_ROOT_KEY)
        if base_url is None:
            base_url = "https://edge.nengo.ai/gql"
        self.base_url = base_url
        self.headers = custom_headers or {}

        self.set_token(token)

    def set_token(self, new_token: str) -> None:
        """Update the token with a new token (e.g. if the token has expired)."""
        self.headers["Authorization"] = f"Bearer {new_token}"
        transport = RequestsHTTPTransport(
            url=self.base_url,
            headers=self.headers,
        )
        self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def _execute_gql(self, gql_string: str, variables: Optional[Dict] = None) -> Dict:
        """Execute the given gql query with optional variables."""
        query_or_mutation = gql(gql_string)
        # Wrong Permissions: TransportQueryError
        try:
            return self.client.execute(query_or_mutation, variable_values=variables)
        except TransportQueryError as e:
            if e.errors and len(e.errors) > 0 and e.errors[0].get("message"):
                raise AuthenticationError(e.errors[0]["message"])
            # Other reasons for invalid token?
            else:
                raise RequestError() from e
        except GraphQLError as e:
            raise RequestError("Invalid GraphQL Query") from e

    def get_projects(self) -> List[Dict[str, Any]]:
        """Returns a list of project ids and names."""
        response = self._execute_gql(
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

    def get_datasets(self, model_type: str) -> List[Dict[str, Any]]:
        """Get a list of datasets for the given model type."""

        get_base_data_query = """
        query GetBasedatas($modelType: ModelType) {
            baseData(modelType: $modelType) {
            datasets {
                _id
                name
                }
            }
        }
        """

        baseData = self._execute_gql(get_base_data_query, {"modelType": model_type})[
            "baseData"
        ]

        datasets = []
        for base in baseData:
            datasets.extend(base["datasets"])
        return datasets

    def get_networks(self, model_type: str) -> List[Dict[str, Any]]:
        """Get a list of networks for the given model type."""

        get_networks_query = """
         query GetAllNetworks($modelType: ModelType!) {
            networks(modelType: $modelType) {
            _id
            name
            }
        }
        """
        return self._execute_gql(get_networks_query, {"modelType": model_type})[
            "networks"
        ]

    def add_run(self, project_id: str, model_type: str) -> Dict[str, Any]:
        """Create a new run."""
        add_run_mutation = """
        mutation AddRun($id: ObjectId!, $modelType: ModelType!) {
            addRun(project: $id, modelType: $modelType) {
            _id
            }
        }
        """

        return self._execute_gql(
            add_run_mutation, {"id": project_id, "modelType": model_type}
        )["addRun"]

    def update_run(self, run_id: str, run_input: Dict[str, Any]) -> Dict[str, Any]:
        """Update a run."""
        update_run_mutation = """
        mutation UpdateRun($id: ObjectId!, $runInput: RunInput!) {
            updateRun(run: $id, updates: $runInput) {
            _id
            }
        }
        """
        return self._execute_gql(
            update_run_mutation,
            variables={"id": run_id, "runInput": run_input},
        )["updateRun"]

    def start_optimize(self, run_id: str) -> dict:
        """Start optimizing a run."""
        start_optimize_mutation = """
            mutation StartOptimize($id: ObjectId!) {
                startOptimize(run: $id) {
                _id
                }
            }
        """
        return self._execute_gql(start_optimize_mutation, variables={"id": run_id})[
            "startOptimize"
        ]

    def get_results(self, run_id: str) -> Dict:
        """Get the run status along with results if present."""
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
        response = self._execute_gql(get_results_query, variables={"id": run_id})
        return response["run"]

    # Higher level functions
    def start_training_run(
        self,
        project_id: str,
        model_type: str,
        hyperparams: Dict[str, Any],
        hardware: str = "gpu",
        dataset: Optional[str] = None,
        network: Optional[str] = None,
    ) -> str:
        """Create a run with the given model type and hyperparameters, and start
        training it."""
        add_response = self.add_run(project_id, model_type)
        run_id = add_response["_id"]
        run_input: Dict[str, Any] = {"hyperparams": hyperparams}
        run_input["hardwareId"] = hardware
        if dataset is not None:
            run_input["datasetId"] = dataset
        if network is not None:
            run_input["networkId"] = network
        self.update_run(run_id, run_input)
        self.start_optimize(run_id)
        return run_id
