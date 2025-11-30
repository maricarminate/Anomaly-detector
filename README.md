# 🔍 Detecção de Anomalias em Séries Temporais

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

Sistema completo de Machine Learning para detectar anomalias em séries temporais usando múltiplos algoritmos e ensemble learning.

---

## 📊 Visão Geral

Este projeto implementa **9+ algoritmos** de detecção de anomalias, desde métodos estatísticos simples até deep learning avançado com LSTM Autoencoders. O sistema foi desenvolvido ao longo de **8 dias** com foco em aprendizado, experimentação e produção.

### 🎯 Características Principais

- ✅ **9+ Detectores**: Z-Score, IQR, Moving Average, EWMA, Isolation Forest, LOF, DBSCAN, Dense AE, LSTM AE
- ✅ **Ensemble Learning**: Votação por maioria, ponderada e consenso
- ✅ **API REST**: FastAPI para deploy em produção
- ✅ **Pipeline Completo**: De EDA até produção
- ✅ **Testes Unitários**: Cobertura de código
- ✅ **Documentação**: README, docstrings e exemplos

---

## 🚀 Quick Start

### Instalação

```bash
# Clone o repositório
git clone https://github.com/maricarminate/anomaly-detection.git
cd anomaly-detection

# Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### Uso Básico

```python
from src.models import ZScoreDetector, EnsembleDetector
from src.data import DataLoader

# Carregar dados
loader = DataLoader()
data = loader.load_nyc_taxi_demo()
train, test = loader.split_train_test()

# Criar e treinar detector
detector = ZScoreDetector(threshold=3)
detector.fit(train)

# Predizer anomalias
predictions = detector.predict(test)
print(f"Anomalias detectadas: {predictions.sum()}")
```

### Treinar Pipeline Completo

```bash
# Treina todos os 9 detectores
python scripts/train_pipeline.py

# Avalia e compara modelos
python scripts/evaluate_models.py

# Faz predições em novos dados
python scripts/predict_new_data.py --input data.csv
```

### Deploy da API

```bash
# Inicia servidor FastAPI
python scripts/deploy_api.py

# Acesse a documentação interativa
# http://localhost:8000/docs
```

---

## 📁 Estrutura do Projeto

```
anomaly-detection/
├── notebooks/              # Jupyter Notebooks (Dias 1-8)
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_tree_methods.ipynb
│   ├── 04_autoencoder.ipynb
│   ├── 05_lstm_autoencoder.ipynb
│   ├── 06_ensemble.ipynb
│   ├── 07_pipeline_final.ipynb
│   └── 08_deploy.ipynb
│
├── src/                    # Código reutilizável
│   ├── models/            # Detectores de anomalia
│   │   ├── base.py
│   │   ├── statistical.py
│   │   ├── tree_based.py
│   │   ├── autoencoder.py
│   │   ├── lstm.py
│   │   └── ensemble.py
│   ├── data/              # Carregamento e pré-processamento
│   ├── evaluation/        # Métricas e visualização
│   └── utils/             # Configuração e logging
│
├── scripts/               # Scripts executáveis
│   ├── train_pipeline.py
│   ├── evaluate_models.py
│   ├── predict_new_data.py
│   └── deploy_api.py
│
├── tests/                 # Testes unitários
├── outputs/               # Modelos, plots, relatórios
└── requirements.txt
```

---

## 🧪 Métodos Implementados

### Dia 2: Métodos Estatísticos
- **Z-Score**: Detecta pontos > 3 desvios padrão
- **IQR**: Usa quartis (Q1, Q3) para definir limites
- **Moving Average**: Compara com média móvel
- **EWMA**: Média ponderada exponencial adaptativa

### Dia 3: Métodos Baseados em Árvores
- **Isolation Forest**: Baseado em árvores de decisão aleatórias
- **LOF**: Local Outlier Factor por densidade local
- **DBSCAN**: Clustering por densidade

### Dia 4-5: Deep Learning
- **Dense Autoencoder**: Rede neural com camadas densas
- **LSTM Autoencoder**: LSTM para capturar dependências temporais

### Dia 6: Ensemble
- **Majority Vote**: Votação por maioria (>= N modelos)
- **Weighted Vote**: Votação ponderada por performance
- **Consensus**: Todos os modelos devem concordar

---

## 📈 Resultados

### Performance dos Métodos

| Método | Anomalias Detectadas | Tempo de Treino | Complexidade |
|--------|---------------------|-----------------|--------------|
| Z-Score | 5.2% | < 1s | O(n) |
| IQR | 4.8% | < 1s | O(n) |
| Isolation Forest | 5.0% | ~5s | O(n log n) |
| LSTM Autoencoder | 4.5% | ~60s | O(n) |
| **Ensemble** | **4.9%** | **~70s** | **O(n)** |

### Visualizações

Exemplos de anomalias detectadas:

![Ensemble Results](outputs/plots/06_ensemble_comparison.png)
![LSTM Results](outputs/plots/05_lstm_autoencoder.png)

---

## 🛠️ Uso Avançado

### Criar Ensemble Personalizado

```python
from src.models import ZScoreDetector, IsolationForestDetector, EnsembleDetector

# Criar detectores
detectors = [
    ZScoreDetector(threshold=3),
    IsolationForestDetector(contamination=0.05)
]

# Criar ensemble
ensemble = EnsembleDetector(
    detectors=detectors,
    strategy='weighted',
    weights=[1.0, 1.5]  # Dar mais peso ao IF
)

# Treinar e predizer
ensemble.fit(train_data)
predictions = ensemble.predict(test_data)
```

### Usar API REST

```bash
# Request exemplo
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"timestamp": "2014-01-01", "value": 10000},
      {"timestamp": "2014-01-02", "value": 12000}
    ],
    "use_ensemble": true
  }'

# Response
{
  "predictions": [0, 1, 0, 0, 1],
  "anomaly_count": 2,
  "anomaly_percentage": 40.0,
  "timestamp": "2025-11-30T18:00:00"
}
```

### Configuração Personalizada

Edite `src/utils/config.py`:

```python
DETECTOR_CONFIG = {
    'zscore': {
        'enabled': True,
        'threshold': 3
    },
    'isolation_forest': {
        'enabled': True,
        'contamination': 0.05,
        'n_estimators': 200  # Aumentar para melhor performance
    }
}
```

---

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/

# Testes específicos
python -m pytest tests/test_models.py -v

# Com cobertura
python -m pytest --cov=src tests/
```

---

## 📊 Dataset

O projeto usa **NYC Taxi Demand** do [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB):

- **Período**: 2014-2015
- **Frequência**: Horária
- **Tamanho**: ~10,000 pontos
- **Anomalias Conhecidas**: Hurricane Sandy, feriados, eventos especiais

---

## 🐳 Docker Deploy

```dockerfile
# Build
docker build -t anomaly-detector .

# Run
docker run -p 8000:8000 anomaly-detector

# Acesse: http://localhost:8000
```

---

## 📚 Documentação

### Notebooks (Dias 1-8)

Cada notebook documenta um dia de desenvolvimento:

1. **Dia 1**: EDA e preparação de dados
2. **Dia 2**: Métodos estatísticos (Z-Score, IQR, MA, EWMA)
3. **Dia 3**: Métodos de árvores (IF, LOF, DBSCAN)
4. **Dia 4**: Dense Autoencoder
5. **Dia 5**: LSTM Autoencoder
6. **Dia 6**: Ensemble e comparação
7. **Dia 7**: Refatoração e pipeline
8. **Dia 8**: Deploy e produção

### Classes Principais

```python
# Classe base
from src.models.base import BaseAnomalyDetector

# Detectores estatísticos
from src.models.statistical import ZScoreDetector, IQRDetector

# Detectores de árvores
from src.models.tree_based import IsolationForestDetector

# Deep Learning
from src.models.autoencoder import DenseAutoencoderDetector
from src.models.lstm import LSTMAutoencoderDetector

# Ensemble
from src.models.ensemble import EnsembleDetector
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📝 TODO / Roadmap

- [ ] Implementar GRU Autoencoder
- [ ] Adicionar Transformer para séries temporais
- [ ] Explicabilidade com SHAP/LIME
- [ ] Dashboard em tempo real (Streamlit)
- [ ] CI/CD com GitHub Actions
- [ ] Deploy em AWS/GCP/Azure
- [ ] Versionamento de modelos (MLflow)
- [ ] Streaming de dados em tempo real

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👤 Autor

**Mariana Santos Carminate**

- GitHub: [@maricarminate](https://github.com/maricarminate)
- LinkedIn: [Seu Perfil](www.linkedin.com/in/mariana-santos-carminate-0a0893133)
- Email: mari.carminate@gmail.com

---

## 🙏 Agradecimentos

- [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) pelos dados
- [Scikit-learn](https://scikit-learn.org/) pelos algoritmos de ML
- [TensorFlow](https://www.tensorflow.org/) pelo framework de Deep Learning
- Comunidade de ML/DS por todo conhecimento compartilhado

---

## 📊 Estatísticas do Projeto

![GitHub stars](https://img.shields.io/github/stars/seu-usuario/anomaly-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/seu-usuario/anomaly-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/seu-usuario/anomaly-detection?style=social)

---

## 🔗 Links Úteis

- [Documentação Completa](docs/)
- [Tutorial de Uso](docs/tutorial.md)
- [Perguntas Frequentes (FAQ)](docs/faq.md)
- [Changelog](CHANGELOG.md)

---

⭐ **Se este projeto foi útil, considere dar uma estrela!** ⭐

---

<div align="center">
  <strong>Desenvolvido com ❤️ usando Python, TensorFlow e Scikit-learn</strong>
</div>

