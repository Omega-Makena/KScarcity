"""Domain management API endpoints - v2."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager
from app.core.domain_manager import DistributionType, DomainStatus

router = APIRouter()


class DomainCreate(BaseModel):
    """Domain creation request."""
    name: Optional[str] = None
    distribution_type: DistributionType = DistributionType.NORMAL
    distribution_params: Optional[Dict[str, float]] = None


class DomainUpdate(BaseModel):
    """Domain update request."""
    distribution_params: Optional[Dict[str, float]] = None
    synthetic_enabled: Optional[bool] = None


class DomainInfo(BaseModel):
    """Domain information response."""
    id: int
    name: str
    status: str
    distribution_type: str
    total_windows: int
    manual_uploads: int
    federation_rounds: int
    last_data_at: Optional[str]
    created_at: str
    synthetic_enabled: bool


@router.post("", response_model=DomainInfo, status_code=201)
async def create_domain(
    request: DomainCreate,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DomainInfo:
    """
    Create a new domain.
    
    Creates a named domain with specified data distribution.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    try:
        domain = scarcity.domain_manager.create_domain(
            name=request.name,
            distribution_type=request.distribution_type,
            distribution_params=request.distribution_params
        )
        
        # Start generation if multi-domain generator is running
        if scarcity.multi_domain_generator and scarcity.multi_domain_generator.running:
            num_domains = len(scarcity.domain_manager.list_domains())
            import asyncio
            asyncio.create_task(
                scarcity.multi_domain_generator._start_domain_generation(domain, num_domains)
            )
        
        return DomainInfo(
            id=domain.id,
            name=domain.name,
            status=domain.status.value,
            distribution_type=domain.distribution_type.value,
            total_windows=domain.total_windows,
            manual_uploads=domain.manual_uploads,
            federation_rounds=domain.federation_rounds,
            last_data_at=domain.last_data_at.isoformat() + "Z" if domain.last_data_at else None,
            created_at=domain.created_at.isoformat() + "Z",
            synthetic_enabled=domain.synthetic_enabled
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[DomainInfo])
async def list_domains(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[DomainInfo]:
    """
    List all domains.
    
    Returns all domains with their current status.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    domains = scarcity.domain_manager.list_domains()
    
    return [
        DomainInfo(
            id=domain.id,
            name=domain.name,
            status=domain.status.value,
            distribution_type=domain.distribution_type.value,
            total_windows=domain.total_windows,
            manual_uploads=domain.manual_uploads,
            federation_rounds=domain.federation_rounds,
            last_data_at=domain.last_data_at.isoformat() + "Z" if domain.last_data_at else None,
            created_at=domain.created_at.isoformat() + "Z",
            synthetic_enabled=domain.synthetic_enabled
        )
        for domain in domains
    ]


@router.get("/{domain_id}", response_model=DomainInfo)
async def get_domain(
    domain_id: int,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DomainInfo:
    """
    Get domain details.
    
    Returns detailed information about a specific domain.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    domain = scarcity.domain_manager.get_domain(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
    
    return DomainInfo(
        id=domain.id,
        name=domain.name,
        status=domain.status.value,
        distribution_type=domain.distribution_type.value,
        total_windows=domain.total_windows,
        manual_uploads=domain.manual_uploads,
        federation_rounds=domain.federation_rounds,
        last_data_at=domain.last_data_at.isoformat() + "Z" if domain.last_data_at else None,
        created_at=domain.created_at.isoformat() + "Z",
        synthetic_enabled=domain.synthetic_enabled
    )


@router.patch("/{domain_id}", response_model=DomainInfo)
async def update_domain(
    domain_id: int,
    request: DomainUpdate,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DomainInfo:
    """
    Update domain configuration.
    
    Updates domain settings like distribution parameters.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    domain = scarcity.domain_manager.get_domain(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
    
    # Update fields
    if request.distribution_params is not None:
        domain.distribution_params = request.distribution_params
        # Recreate generator with new params
        if scarcity.multi_domain_generator and domain_id in scarcity.multi_domain_generator.generators:
            from app.core.multi_domain_generator import DataGenerator
            scarcity.multi_domain_generator.generators[domain_id] = DataGenerator(
                domain.distribution_type,
                domain.distribution_params,
                scarcity.multi_domain_generator.features
            )
    
    if request.synthetic_enabled is not None:
        domain.synthetic_enabled = request.synthetic_enabled
        if scarcity.multi_domain_generator:
            if request.synthetic_enabled and domain.status == DomainStatus.ACTIVE:
                scarcity.multi_domain_generator.resume_domain(domain_id)
            else:
                scarcity.multi_domain_generator.pause_domain(domain_id)
    
    return DomainInfo(
        id=domain.id,
        name=domain.name,
        status=domain.status.value,
        distribution_type=domain.distribution_type.value,
        total_windows=domain.total_windows,
        manual_uploads=domain.manual_uploads,
        federation_rounds=domain.federation_rounds,
        last_data_at=domain.last_data_at.isoformat() + "Z" if domain.last_data_at else None,
        created_at=domain.created_at.isoformat() + "Z",
        synthetic_enabled=domain.synthetic_enabled
    )


@router.delete("/{domain_id}", status_code=204)
async def delete_domain(
    domain_id: int,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
):
    """
    Remove domain.
    
    Removes domain and cleans up resources.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    try:
        # Stop generation first
        if scarcity.multi_domain_generator:
            scarcity.multi_domain_generator.pause_domain(domain_id)
        
        # Remove domain
        scarcity.domain_manager.remove_domain(domain_id)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{domain_id}/pause", response_model=DomainInfo)
async def pause_domain(
    domain_id: int,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DomainInfo:
    """
    Pause domain.
    
    Pauses synthetic data generation for domain.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    try:
        scarcity.domain_manager.pause_domain(domain_id)
        
        # Stop generation
        if scarcity.multi_domain_generator:
            scarcity.multi_domain_generator.pause_domain(domain_id)
        
        domain = scarcity.domain_manager.get_domain(domain_id)
        
        return DomainInfo(
            id=domain.id,
            name=domain.name,
            status=domain.status.value,
            distribution_type=domain.distribution_type.value,
            total_windows=domain.total_windows,
            manual_uploads=domain.manual_uploads,
            federation_rounds=domain.federation_rounds,
            last_data_at=domain.last_data_at.isoformat() + "Z" if domain.last_data_at else None,
            created_at=domain.created_at.isoformat() + "Z",
            synthetic_enabled=domain.synthetic_enabled
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{domain_id}/resume", response_model=DomainInfo)
async def resume_domain(
    domain_id: int,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DomainInfo:
    """
    Resume domain.
    
    Resumes synthetic data generation for domain.
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    try:
        scarcity.domain_manager.resume_domain(domain_id)
        
        # Resume generation
        if scarcity.multi_domain_generator:
            scarcity.multi_domain_generator.resume_domain(domain_id)
        
        domain = scarcity.domain_manager.get_domain(domain_id)
        
        return DomainInfo(
            id=domain.id,
            name=domain.name,
            status=domain.status.value,
            distribution_type=domain.distribution_type.value,
            total_windows=domain.total_windows,
            manual_uploads=domain.manual_uploads,
            federation_rounds=domain.federation_rounds,
            last_data_at=domain.last_data_at.isoformat() + "Z" if domain.last_data_at else None,
            created_at=domain.created_at.isoformat() + "Z",
            synthetic_enabled=domain.synthetic_enabled
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class UploadResponse(BaseModel):
    """Upload response."""
    domain_id: int
    rows_processed: int
    windows_created: int
    message: str


@router.post("/{domain_id}/upload", response_model=UploadResponse)
async def upload_data(
    domain_id: int,
    file: UploadFile = File(...),
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> UploadResponse:
    """
    Upload CSV data to domain.
    
    Accepts CSV file with scarcity data and assigns it to the specified domain.
    Expected columns: timestamp, feature1, feature2, ..., scarcity_signal
    """
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    # Verify domain exists
    domain = scarcity.domain_manager.get_domain(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_cols = ['timestamp', 'scarcity_signal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract feature columns (all except timestamp and scarcity_signal)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'scarcity_signal']]
        
        if not feature_cols:
            raise HTTPException(
                status_code=400,
                detail="No feature columns found. CSV must contain at least one feature column."
            )
        
        # Process data into windows
        windows_created = 0
        
        # Group by timestamp to create windows
        for timestamp, group in df.groupby('timestamp'):
            # Extract features and scarcity signal
            features = group[feature_cols].values
            scarcity_signal = group['scarcity_signal'].values[0]
            
            # Publish to event bus (will be captured by DomainDataStore)
            if scarcity.bus:
                import asyncio
                asyncio.create_task(scarcity.bus.publish("data_window", {
                    "data": features,
                    "domain_id": domain_id,
                    "domain_name": domain.name,
                    "window_id": domain.total_windows + windows_created + 1,
                    "timestamp": timestamp.isoformat() + "Z",
                    "source": "manual",
                    "scarcity_signal": float(scarcity_signal),
                    "upload_id": domain.manual_uploads + 1
                }))
            
            windows_created += 1
        
        # Update domain statistics
        domain.manual_uploads += 1
        domain.total_windows += windows_created
        domain.last_data_at = datetime.utcnow()
        
        return UploadResponse(
            domain_id=domain_id,
            rows_processed=len(df),
            windows_created=windows_created,
            message=f"Successfully uploaded {len(df)} rows, created {windows_created} windows"
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
