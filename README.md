# 🤖 GPT Models Comparison Dashboard

A comprehensive Streamlit application for comparing 30+ GPT and large language models with detailed specifications, memory requirements, and hardware recommendations.

## ✨ Features

- 📊 **Comprehensive Model Database**: 30+ models including GPT-4o, GPT-3.5, LLaMA 2, Falcon, Mistral, and more
- 🔍 **Advanced Filtering**: Filter by provider, memory, parameters, and model selection
- 📈 **Interactive Visualizations**: 
  - Parameters vs Memory scatter plots
  - Inference speed comparisons
  - Context window analysis
  - GPU memory requirements
  - Multi-metric performance scoring
- 🎯 **Detailed Specifications**: In-depth info for each model with expandable cards
- ⚙️ **Hardware Recommendations**: GPU tiers, memory bandwidth, and server configurations
- 🏢 **Provider Comparison**: Compare models across OpenAI, Meta, Mistral, TII, and EleutherAI

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gpt-models-comparison.git
cd gpt-models-comparison
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📋 Models Included

### OpenAI Models
- GPT-4o (120B parameters)
- GPT-4o mini (4B parameters)
- GPT-4 Turbo (175B parameters)
- GPT-4 (100B parameters)
- GPT-3.5 Turbo (15B parameters)
- GPT-3.5 (175B parameters)
- Legacy models (text-davinci-003, text-davinci-002, etc.)

### Open Source Models
- **Meta LLaMA 2**: 7B, 13B, 70B
- **Mistral**: 7B, 8x7B
- **TII Falcon**: 7B, 40B, 180B
- **EleutherAI**: GPT-Neo (2.7B, 20B), GPT-J (6B)

### Historical Models
- GPT-2 (4 variants: Large, Medium, Small, Distilled)
- GPT-1 (117M parameters)

## 🎨 Dashboard Features

### Tab 1: Comparison Table
- Side-by-side comparison of all models
- Sortable columns with formatted data
- Quick filtering by model selection
- Memory and parameter range filters

### Tab 2: Visualizations
- **Parameters vs Memory**: Scatter plot showing model size relationships
- **Inference Speed**: Bar chart comparing tokens/second throughput
- **Context Window**: Model context length comparison
- **GPU Memory**: VRAM requirements visualization
- **Multi-Metric Score**: Normalized performance comparison across metrics

### Tab 3: Specifications
- Expandable accordion cards for each model
- Complete technical specifications:
  - Model parameters
  - Training data size
  - Context window
  - Memory requirements (CPU & GPU)
  - Inference speed
  - Recommended GPU hardware
  - Release date and type

### Tab 4: Hardware Requirements
- **GPU Specifications Table**:
  - RTX 3080, 4090, A100, H100, L40S
  - Memory bandwidth comparison
  - Computational performance (TFLOPS)
  - Power consumption
  - Pricing estimates

- **Server Configurations**:
  - Light Models (GPT-3.5 Turbo)
  - Medium Models (GPT-4)
  - Large Models (GPT-4o, Falcon 180B)
  - Cost estimates for each tier

- **Power Consumption Analysis**: GPU power requirements visualization

## 🔧 Sidebar Filters

- **Provider Filter**: Select models from specific organizations (OpenAI, Meta, Mistral, TII, EleutherAI)
- **Model Selection**: Choose which models to compare
- **Memory Range**: Filter by RAM requirements (0-400+ GB)
- **Parameter Range**: Filter by model size (0-200+ billion parameters)

## 📊 Data Structure

Each model includes:
- Model name and type
- Parameters (in billions)
- Context window size
- Training data size (GB)
- Total memory requirement (GB)
- GPU memory requirement (GB)
- Inference speed (tokens/second)
- Release date
- Model type/category
- Max batch size
- Recommended GPU(s)
- Provider/Organization

## 🛠️ Technology Stack

- **Streamlit**: Interactive web framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computations
- **Matplotlib**: Additional plotting capabilities

## 📦 Requirements

```
streamlit==1.31.1
pandas==2.1.3
plotly==5.18.0
numpy==1.24.3
matplotlib==3.8.2
```

## 🔄 Updating Model Data

To add or update models, edit the `models_data` dictionary in `app.py`:

```python
models_data = {
    "Model": [...],
    "Parameters (Billions)": [...],
    "Context Window": [...],
    # ... other fields
}
```

## 💾 Project Structure

```
gpt-models-comparison/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore file
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. **Fork or clone this repository to your GitHub account**

2. **Deploy with Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select this repository
   - Choose `app.py` as the main file

### Deploy to Heroku

```bash
heroku create your-app-name
git push heroku main
```

### Deploy to Other Platforms
- AWS EC2
- Google Cloud Run
- Azure App Service
- Digital Ocean

## 📈 Performance Notes

- The app handles 30+ models smoothly
- All visualizations are interactive and responsive
- Filters are applied in real-time
- Memory usage is minimal even with large datasets

## 🐛 Troubleshooting

**Port Already in Use**:
```bash
streamlit run app.py --server.port 8502
```

**Cache Issues**:
```bash
streamlit cache clear
streamlit run app.py
```

**Dependencies Not Found**:
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Notes

- All specifications are based on official documentation and benchmarks
- Actual performance may vary based on hardware and implementation
- GPU prices and power consumption are approximate
- Recommended configurations are guidelines that may need adjustment based on use cases
- Data updated as of December 2024

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Created with ❤️ by [Your Name](https://github.com/yourusername)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: your.email@example.com

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI Models](https://platform.openai.com/docs/models)
- [Meta LLaMA](https://github.com/facebookresearch/llama)
- [Mistral AI](https://mistral.ai/)
- [Hugging Face Model Hub](https://huggingface.co/models)

---

**Last Updated**: December 3, 2024

