"""
Script: API FastAPI para Detecção de Anomalias
Deploy em produção com endpoint REST

Instalar: pip install fastapi uvicorn

Execute: python scripts/deploy_api.py
Ou: uvicorn scripts.deploy_api:app --reload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from glob import glob

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI não instalado")
    print("Execute: pip install fastapi uvicorn")
    sys.exit(1)

from src.models import BaseAnomalyDetector
from src.utils import MODELS_DIR


# ============================================================================
# MODELOS DE DADOS
# ============================================================================

class TimeSeriesPoint(BaseModel):
    """Ponto de série temporal"""
    timestamp: Optional[str] = None
    value: float


class PredictionRequest(BaseModel):
    """Request para predição"""
    data: List[TimeSeriesPoint]
    use_ensemble: bool = True


class PredictionResponse(BaseModel):
    """Response de predição"""
    predictions: List[int]
    anomaly_count: int
    anomaly_percentage: float
    timestamp: str


class ModelInfo(BaseModel):
    """Informações sobre modelos disponíveis"""
    available_models: List[str]
    ensemble_available: bool
    total_models: int


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Anomaly Detection API",
    description="API para detecção de anomalias em séries temporais",
    version="1.0.0"
)


# Cache de modelos (carregados uma vez)
_models_cache = {}


def load_models():
    """Carrega modelos em cache"""
    if _models_cache:
        return _models_cache
    
    model_files = glob(str(MODELS_DIR / "*.pkl"))
    
    for model_file in model_files:
        name = Path(model_file).stem
        try:
            _models_cache[name] = BaseAnomalyDetector.load(model_file)
        except Exception as e:
            print(f"Erro ao carregar {name}: {e}")
    
    return _models_cache


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Carrega modelos ao iniciar"""
    print("Carregando modelos...")
    models = load_models()
    print(f"✓ {len(models)} modelos carregados")


@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Detectar anomalias",
            "/models": "GET - Listar modelos disponíveis",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    models = load_models()
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models", response_model=ModelInfo)
async def list_models():
    """Lista modelos disponíveis"""
    models = load_models()
    
    ensemble_files = [k for k in models.keys() if 'ensemble' in k.lower()]
    
    return ModelInfo(
        available_models=list(models.keys()),
        ensemble_available=len(ensemble_files) > 0,
        total_models=len(models)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Detecta anomalias em série temporal
    
    Args:
        request: PredictionRequest com dados e configuração
    
    Returns:
        PredictionResponse com predições
    """
    
    # Validar entrada
    if not request.data:
        raise HTTPException(status_code=400, detail="Dados vazios")
    
    if len(request.data) < 10:
        raise HTTPException(status_code=400, detail="Mínimo 10 pontos necessários")
    
    # Converter para DataFrame
    df = pd.DataFrame([{
        'timestamp': p.timestamp,
        'value': p.value
    } for p in request.data])
    
    # Carregar modelos
    models = load_models()
    
    if not models:
        raise HTTPException(status_code=500, detail="Nenhum modelo disponível")
    
    # Selecionar modelo
    if request.use_ensemble:
        ensemble_models = {k: v for k, v in models.items() if 'ensemble' in k.lower()}
        
        if not ensemble_models:
            raise HTTPException(status_code=404, detail="Ensemble não disponível")
        
        # Usar ensemble mais recente
        model_name = sorted(ensemble_models.keys())[-1]
        model = ensemble_models[model_name]
    else:
        # Usar primeiro modelo disponível (pode ser melhorado)
        model_name = list(models.keys())[0]
        model = models[model_name]
    
    # Predizer
    try:
        predictions = model.predict(df)
        predictions_list = predictions.tolist()
        
        # Calcular estatísticas
        anomaly_count = int(np.sum(predictions))
        anomaly_pct = float(100 * anomaly_count / len(predictions))
        
        return PredictionResponse(
            predictions=predictions_list,
            anomaly_count=anomaly_count,
            anomaly_percentage=round(anomaly_pct, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[str]):
    """
    Prediz em múltiplos arquivos (para uso futuro)
    """
    return {"message": "Batch prediction - não implementado ainda"}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Inicia servidor API"""
    
    print("=" * 70)
    print("ANOMALY DETECTION API")
    print("=" * 70)
    
    # Verificar modelos
    models = load_models()
    
    if not models:
        print("\n❌ Nenhum modelo encontrado!")
        print("Execute primeiro: python scripts/train_pipeline.py")
        return
    
    print(f"\n✓ {len(models)} modelos carregados:")
    for name in models.keys():
        print(f"  - {name}")
    
    print("\n🚀 Iniciando servidor...")
    print("\n📡 API disponível em:")
    print("  http://localhost:8000")
    print("  http://localhost:8000/docs (Swagger UI)")
    print("\n⌨️  Pressione Ctrl+C para parar\n")
    
    uvicorn.run(
        "scripts.deploy_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == '__main__':
    main()