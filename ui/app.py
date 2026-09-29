def query_backend_api(
    payload: Any,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Execute a query against the FastAPI backend, with graceful fallback to in-process client."""

    if isinstance(payload, str):
        payload = {"query": payload}

    clean_url = api_url.rstrip("/")

    try:
        resp = requests.post(
            f"{clean_url}/query",
            json=payload,
            timeout=60,
        )

        if resp.status_code == 200:
            return resp.json()

        elif resp.status_code == 422:
            return {
                "status": "INVALID_INPUT",
                "query": payload.get("query", ""),
                "answer": "Input validation error: Please verify your query parameters.",
                "warnings": [f"HTTP 422: {resp.text}"],
                "sources": [],
                "citations": [],
                "results": [],
            }

        else:
            return {
                "status": "ERROR",
                "query": payload.get("query", ""),
                "answer": f"Backend API returned status code {resp.status_code}.",
                "warnings": [resp.text],
                "sources": [],
                "citations": [],
                "results": [],
            }

    except Exception as e:
        logger.info(
            "Direct HTTP connection failed (%s); utilizing in-process pipeline client.",
            e,
        )

        try:
            from pydantic import ValidationError
            from api.main import execute_query
            from api.schemas import QueryRequest

            try:
                req = QueryRequest(**payload)
            except (ValueError, ValidationError) as val_err:
                return {
                    "status": "INVALID_INPUT",
                    "query": payload.get("query", ""),
                    "answer": "Input validation error: Please verify your query parameters.",
                    "warnings": [f"HTTP 422: {str(val_err)}"],
                    "sources": [],
                    "citations": [],
                    "results": [],
                }

            res = execute_query(req)
            data = res.model_dump()
            data["_in_process_fallback"] = True
            return data

        except Exception as inner_e:
            logger.error(
                "Failed in-process execution fallback: %s",
                inner_e,
            )
            return {
                "status": "ERROR",
                "query": payload.get("query", ""),
                "answer": f"In-process engine error: {str(inner_e)}.",
                "warnings": [str(inner_e)],
                "sources": [],
                "citations": [],
                "results": [],
            }