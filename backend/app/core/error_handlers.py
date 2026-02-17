"""Error handling middleware and exception handlers."""

import logging
from typing import Union

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Provides consistent error response format.
    """
    logger.warning(
        f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path)
            }
        }
    )


async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors.
    
    Provides detailed validation error information.
    """
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": 422,
                "message": "Validation error",
                "details": exc.errors(),
                "path": str(request.url.path)
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Logs full exception and returns generic error to client.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}",
        exc_info=exc
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "path": str(request.url.path)
            }
        }
    )


class ComponentNotInitializedError(Exception):
    """Raised when a scarcity component is not initialized."""
    pass


class ComponentStartupError(Exception):
    """Raised when a scarcity component fails to start."""
    pass


class DataIngestionError(Exception):
    """Raised when data ingestion fails."""
    pass
