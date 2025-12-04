import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Set page config
st.set_page_config(
    page_title="GPT Model Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 GPT Models Comparison Dashboard")
st.markdown("""
This dashboard provides a comprehensive comparison of various GPT models, 
including their specifications, memory requirements, and hardware details.
""")

# Model data - Comprehensive GPT Models Database
models_data = {
    "Model": [
        "GPT-4o",
        "GPT-4o mini",
        "GPT-4 Turbo",
        "GPT-4 Turbo Vision",
        "GPT-4 (8K)",
        "GPT-4 (32K)",
        "GPT-3.5 Turbo",
        "GPT-3.5 Turbo 16K",
        "GPT-3.5",
        "Text-davinci-003",
        "Text-davinci-002",
        "Text-curie-001",
        "Text-babbage-001",
        "Text-ada-001",
        "GPT-2 (Large)",
        "GPT-2 (Medium)",
        "GPT-2 (Small)",
        "GPT-2 (Distilled)",
        "GPT-1",
        "GPT-Neo 2.7B",
        "GPT-Neo 20B",
        "GPT-J 6B",
        "Falcon 7B",
        "Falcon 40B",
        "Falcon 180B",
        "LLaMA 2 7B",
        "LLaMA 2 13B",
        "LLaMA 2 70B",
        "Mistral 7B",
        "Mixtral 8x7B"
    ],
    "Parameters (Billions)": [
        120, 4, 175, 175, 100, 100, 15, 15, 175, 175, 175, 13, 1.3, 0.35,
        1.5, 0.8, 0.3, 0.12, 0.117, 2.7, 20, 6, 7, 40, 180, 7, 13, 70, 7, 46.7
    ],
    "Context Window": [
        128000, 128000, 128000, 128000, 8192, 32000, 4096, 16000, 4096, 4096, 4096, 
        2048, 2048, 2048, 1024, 1024, 1024, 1024, 512, 2048, 2048, 2048, 2048, 2048, 
        3900, 4096, 4096, 4096, 8192, 32000
    ],
    "Training Data Size (GB)": [
        13000, 8000, 13000, 13000, 13000, 13000, 1300, 1300, 1300, 1300, 1300,
        300, 300, 300, 40, 40, 40, 40, 7, 200, 300, 600, 1500, 1500, 3500,
        2000, 2000, 2000, 1000, 1000
    ],
    "Memory (GB)": [
        240, 8, 350, 350, 200, 200, 30, 50, 350, 350, 350, 50, 10, 5,
        3, 1.8, 0.8, 0.4, 0.5, 5.4, 40, 12, 14, 80, 360, 14, 26, 140, 14, 95
    ],
    "GPU Memory (GB)": [
        80, 4, 90, 90, 80, 80, 12, 20, 90, 90, 90, 20, 6, 3,
        2, 1, 0.5, 0.2, 0.3, 2.7, 40, 12, 9, 80, 180, 9, 16, 70, 9, 60
    ],
    "Inference Speed (tokens/sec)": [
        80000, 150000, 60000, 60000, 50000, 45000, 120000, 100000, 100000, 90000, 85000,
        80000, 75000, 70000, 50000, 45000, 40000, 35000, 30000, 3000, 2500, 4000, 5000, 
        3500, 2000, 6000, 5500, 3000, 8000, 10000
    ],
    "Release Date": [
        "2024-05", "2024-07", "2023-11", "2023-11", "2023-03", "2023-03", "2023-01", 
        "2023-06", "2022-12", "2022-11", "2022-01", "2022-01", "2022-01", "2022-01",
        "2019-02", "2019-02", "2019-02", "2019-02", "2018-06", "2021-03", "2021-05",
        "2021-06", "2023-03", "2023-04", "2023-09", "2023-07", "2023-07", "2023-07",
        "2024-01", "2024-01"
    ],
    "Type": [
        "Latest", "Mini", "Large", "Large", "Large", "Large", "Standard", "Extended",
        "Large", "Legacy", "Legacy", "Medium", "Small", "Tiny",
        "Classic", "Classic", "Classic", "Classic", "Original", "Alternative", "Alternative",
        "Alternative", "Alternative", "Alternative", "Alternative", "Open Source", "Open Source",
        "Open Source", "Open Source", "Open Source"
    ],
    "Max Batch Size": [
        256, 512, 256, 256, 128, 128, 256, 256, 256, 256, 256, 128, 64, 32,
        64, 64, 64, 32, 32, 16, 8, 8, 32, 16, 8, 32, 32, 16, 32, 32
    ],
    "Recommended GPU": [
        "H100 / A100 (Multiple)", "L40S / A100", "H100 / A100 (Multiple)", "H100 / A100 (Multiple)",
        "A100 (2-4x)", "A100 (Multiple)", "A100 / L40S", "A100 / L40S (2x)", "A100 / L40S (Multiple)",
        "A100", "A100", "A100", "L40S", "RTX 4090", "L40S / RTX 4090", "RTX 4090",
        "RTX 3080 / 4090", "RTX 3080", "RTX 3080", "L40S", "A100 / H100",
        "L40S / A100", "L40S", "A100", "H100 (Multiple)", "A100", "A100", "A100 / H100",
        "A100 / L40S", "A100 / H100 (Multiple)"
    ],
    "Provider": [
        "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI",
        "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI", "OpenAI",
        "OpenAI", "OpenAI", "OpenAI", "EleutherAI", "EleutherAI", "EleutherAI",
        "TII", "TII", "TII", "Meta", "Meta", "Meta", "Mistral", "Mistral"
    ],
    "Open Source": [
        "❌ No", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No",
        "❌ No", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No", "✅ Yes", "✅ Yes",
        "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes",
        "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes"
    ],
    "Access Method": [
        "API Only", "API Only", "API Only", "API Only", "API Only", "API Only", "API Only", "API Only",
        "API Only", "API Only", "API Only", "API Only", "API Only", "API Only", 
        "Hugging Face", "Hugging Face", "Hugging Face", "Hugging Face", "GitHub",
        "Hugging Face", "Hugging Face", "Hugging Face", "Hugging Face", "Hugging Face", "Hugging Face",
        "Meta / Hugging Face", "Meta / Hugging Face", "Meta / Hugging Face", "Hugging Face / Mistral AI", "Hugging Face / Mistral AI"
    ],
    "License": [
        "Proprietary", "Proprietary", "Proprietary", "Proprietary", "Proprietary", "Proprietary", "Proprietary", "Proprietary",
        "Proprietary", "Proprietary", "Proprietary", "Proprietary", "Proprietary", "Proprietary",
        "MIT", "MIT", "MIT", "MIT", "MIT", "Apache 2.0", "Apache 2.0", "Apache 2.0",
        "Apache 2.0", "Apache 2.0", "Apache 2.0", "Meta (Commercial Use OK)", "Meta (Commercial Use OK)", "Meta (Commercial Use OK)",
        "Apache 2.0", "Apache 2.0"
    ],
    "Deployment Options": [
        "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API",
        "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API", "OpenAI API",
        "Local / Cloud", "Local / Cloud", "Local / Cloud", "Local / Cloud", "Local / Cloud",
        "Local / Cloud / HF", "Local / Cloud / HF", "Local / Cloud / HF", 
        "Local / Cloud / HF", "Local / Cloud / HF", "Local / Cloud / HF",
        "Local / Cloud / Replicate", "Local / Cloud / Replicate", "Local / Cloud / Replicate",
        "Local / Cloud / HF", "Local / Cloud / HF"
    ]
}

df = pd.DataFrame(models_data)

# Sidebar filters
st.sidebar.header("🔧 Filters & Options")

# Provider filter
selected_providers = st.sidebar.multiselect(
    "Select providers:",
    options=sorted(df["Provider"].unique().tolist()),
    default=sorted(df["Provider"].unique().tolist())
)

# Filter by provider first
df_provider = df[df["Provider"].isin(selected_providers)]

# Model selection
selected_models = st.sidebar.multiselect(
    "Select models to compare:",
    options=df_provider["Model"].tolist(),
    default=df_provider["Model"].tolist()[:3]
)

# Filter by memory
memory_range = st.sidebar.slider(
    "Filter by Memory (GB):",
    min_value=int(df["Memory (GB)"].min()),
    max_value=int(df["Memory (GB)"].max()),
    value=(0, 400),
    step=10
)

# Filter by parameters
param_range = st.sidebar.slider(
    "Filter by Parameters (Billions):",
    min_value=0.1,
    max_value=float(df["Parameters (Billions)"].max()),
    value=(0.0, 200.0),
    step=5.0
)

# Apply filters
filtered_df = df[
    (df["Model"].isin(selected_models)) &
    (df["Provider"].isin(selected_providers)) &
    (df["Memory (GB)"] >= memory_range[0]) &
    (df["Memory (GB)"] <= memory_range[1]) &
    (df["Parameters (Billions)"] >= param_range[0]) &
    (df["Parameters (Billions)"] <= param_range[1])
]

# Display tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Comparison Table", "📈 Visualizations", "🎯 Specifications", "⚙️ Hardware Requirements", "🔓 Open Source Info"])

# Tab 1: Comparison Table
with tab1:
    st.subheader("Model Comparison Table")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parameters (Billions)": st.column_config.NumberColumn(format="%d B"),
            "Memory (GB)": st.column_config.NumberColumn(format="%d GB"),
            "GPU Memory (GB)": st.column_config.NumberColumn(format="%d GB"),
            "Inference Speed (tokens/sec)": st.column_config.NumberColumn(format="%d"),
            "Max Batch Size": st.column_config.NumberColumn(format="%d"),
            "Context Window": st.column_config.NumberColumn(format="%d"),
            "Training Data Size (GB)": st.column_config.NumberColumn(format="%d GB"),
        }
    )

# Tab 2: Visualizations
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameters vs Memory")
        fig1 = px.scatter(
            filtered_df,
            x="Parameters (Billions)",
            y="Memory (GB)",
            size="Context Window",
            color="Type",
            hover_name="Model",
            title="Model Size vs Memory Requirements",
            labels={"Parameters (Billions)": "Parameters (B)", "Memory (GB)": "Total Memory (GB)"}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Inference Speed Comparison")
        fig2 = px.bar(
            filtered_df.sort_values("Inference Speed (tokens/sec)", ascending=True),
            y="Model",
            x="Inference Speed (tokens/sec)",
            color="Parameters (Billions)",
            title="Inference Speed Comparison",
            labels={"Inference Speed (tokens/sec)": "Tokens/Second"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Context Window Size")
        fig3 = px.bar(
            filtered_df.sort_values("Context Window", ascending=True),
            y="Model",
            x="Context Window",
            color="Context Window",
            title="Context Window Comparison",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.subheader("GPU Memory Requirements")
        fig4 = px.bar(
            filtered_df.sort_values("GPU Memory (GB)", ascending=True),
            y="Model",
            x="GPU Memory (GB)",
            color="GPU Memory (GB)",
            title="GPU Memory Requirements",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Additional comparison chart
    st.subheader("Multi-Metric Comparison")
    
    # Normalize metrics for radar chart
    normalized_df = filtered_df.copy()
    normalized_df["Norm_Parameters"] = normalized_df["Parameters (Billions)"] / normalized_df["Parameters (Billions)"].max() * 100
    normalized_df["Norm_Speed"] = normalized_df["Inference Speed (tokens/sec)"] / normalized_df["Inference Speed (tokens/sec)"].max() * 100
    normalized_df["Norm_Context"] = normalized_df["Context Window"] / normalized_df["Context Window"].max() * 100
    normalized_df["Norm_Memory"] = (1 - normalized_df["Memory (GB)"] / normalized_df["Memory (GB)"].max()) * 100  # Lower is better
    
    # Create comparison bar chart
    comparison_metrics = normalized_df[["Model", "Norm_Parameters", "Norm_Speed", "Norm_Context", "Norm_Memory"]]
    comparison_metrics = comparison_metrics.set_index("Model")
    comparison_metrics.columns = ["Parameters Score", "Speed Score", "Context Score", "Efficiency Score"]
    
    fig5 = px.bar(
        comparison_metrics.reset_index().melt(id_vars="Model"),
        x="Model",
        y="value",
        color="variable",
        title="Normalized Performance Metrics",
        labels={"value": "Score (0-100)", "variable": "Metric"},
        barmode="group"
    )
    st.plotly_chart(fig5, use_container_width=True)

# Tab 3: Specifications
with tab3:
    st.subheader("Detailed Model Specifications")
    
    for idx, row in filtered_df.iterrows():
        with st.expander(f"🔹 {row['Model']} ({row['Type']}) - {row['Release Date']}", expanded=(idx == 0)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Parameters", f"{row['Parameters (Billions)']:.1f}B")
                st.metric("Training Data", f"{row['Training Data Size (GB)']:.0f} GB")
                st.metric("Context Window", f"{row['Context Window']:,}")
            
            with col2:
                st.metric("Total Memory", f"{row['Memory (GB)']:.0f} GB")
                st.metric("GPU Memory", f"{row['GPU Memory (GB)']:.0f} GB")
                st.metric("Batch Size", f"{row['Max Batch Size']}")
            
            with col3:
                st.metric("Inference Speed", f"{row['Inference Speed (tokens/sec)']:,.0f} tokens/s")
                st.info(f"**Recommended GPU:** {row['Recommended GPU']}")

# Tab 4: Hardware Requirements
with tab4:
    st.subheader("Hardware Recommendations by Model")
    
    # GPU comparison
    st.markdown("### GPU Hardware Tiers")
    
    gpu_specs = {
        "GPU": [
            "RTX 3080",
            "RTX 4090",
            "A100 (40GB)",
            "A100 (80GB)",
            "H100",
            "L40S"
        ],
        "VRAM (GB)": [10, 24, 40, 80, 80, 48],
        "Memory Bandwidth (GB/s)": [760, 1008, 2039, 2039, 3456, 864],
        "Peak FP32 (TFLOPS)": [29.3, 82.6, 312, 312, 989, 568],
        "Price ($)": [1499, 1999, 12000, 15000, 40000, 48000],
        "Power (Watts)": [320, 450, 400, 400, 700, 290]
    }
    
    gpu_df = pd.DataFrame(gpu_specs)
    st.dataframe(gpu_df, use_container_width=True, hide_index=True)
    
    # Recommended configurations
    st.markdown("### Recommended Server Configurations")
    
    config1, config2, config3 = st.columns(3)
    
    with config1:
        st.info("""
        **Light Models (GPT-3.5 Turbo)**
        - GPU: Single A100/L40S
        - CPU: 16-core Intel Xeon
        - RAM: 64 GB
        - Storage: 256 GB SSD
        - Estimated Cost: $15,000-20,000
        """)
    
    with config2:
        st.warning("""
        **Medium Models (GPT-4)**
        - GPU: 2x A100 (80GB)
        - CPU: 32-core Intel Xeon
        - RAM: 256 GB
        - Storage: 2 TB NVMe SSD
        - Estimated Cost: $50,000-70,000
        """)
    
    with config3:
        st.error("""
        **Large Models (GPT-4o)**
        - GPU: 4-8x H100 / Multiple A100s
        - CPU: 64-core EPYC/Xeon
        - RAM: 1 TB+
        - Storage: 10 TB NVMe RAID
        - Estimated Cost: $200,000+
        """)
    
    # Power consumption chart
    st.markdown("### Power Consumption Comparison")
    
    power_fig = px.bar(
        gpu_df,
        x="GPU",
        y="Power (Watts)",
        color="Power (Watts)",
        title="GPU Power Consumption",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(power_fig, use_container_width=True)

# Tab 5: Open Source Information
with tab5:
    st.subheader("Open Source Models Information")
    
    # Filter to show only open source models
    open_source_df = filtered_df[filtered_df["Open Source"] == "✅ Yes"].copy()
    
    if len(open_source_df) > 0:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Open Source Models", len(open_source_df))
        with col2:
            st.metric("Avg Parameters", f"{open_source_df['Parameters (Billions)'].mean():.1f}B")
        with col3:
            st.metric("Avg Memory", f"{open_source_df['Memory (GB)'].mean():.0f} GB")
        with col4:
            st.metric("Providers", open_source_df["Provider"].nunique())
        
        st.markdown("---")
        
        # Open Source Model Details
        st.markdown("### Detailed Open Source Model Information")
        
        for idx, row in open_source_df.iterrows():
            with st.expander(f"📖 {row['Model']} - {row['Provider']}", expanded=(idx == open_source_df.index[0])):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**License:** {row['License']}")
                    st.markdown(f"**Open Source:** {row['Open Source']}")
                    st.markdown(f"**Provider:** {row['Provider']}")
                    st.markdown(f"**Release Date:** {row['Release Date']}")
                    st.markdown(f"**Parameters:** {row['Parameters (Billions)']:.1f}B")
                
                with col2:
                    st.markdown(f"**Access Method:** {row['Access Method']}")
                    st.markdown(f"**Deployment Options:** {row['Deployment Options']}")
                    st.markdown(f"**Memory Required:** {row['Memory (GB)']:.0f} GB")
                    st.markdown(f"**GPU Memory:** {row['GPU Memory (GB)']:.0f} GB")
                    st.markdown(f"**Context Window:** {row['Context Window']:,}")
                
                # Access and usage instructions
                st.markdown("#### 🔗 How to Access & Use")
                
                if "Hugging Face" in row['Access Method']:
                    st.markdown("""
                    **Via Hugging Face:**
                    1. Visit [Hugging Face Model Hub](https://huggingface.co/models)
                    2. Search for the model name
                    3. Download model weights or use via API
                    ```python
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    tokenizer = AutoTokenizer.from_pretrained("model-name")
                    model = AutoModelForCausalLM.from_pretrained("model-name")
                    ```
                    """)
                
                if row['Provider'] == "Meta":
                    st.markdown("""
                    **Via Meta:**
                    1. Visit [Meta's LLaMA GitHub](https://github.com/facebookresearch/llama)
                    2. Request access or download directly
                    3. Follow setup instructions in repository
                    ```bash
                    git clone https://github.com/facebookresearch/llama.git
                    ```
                    """)
                
                if row['Provider'] == "Mistral":
                    st.markdown("""
                    **Via Mistral AI:**
                    1. Visit [Mistral AI](https://mistral.ai/)
                    2. Download models from their platform or Hugging Face
                    3. Use with vLLM or Ollama for local inference
                    ```bash
                    ollama run mistral
                    ```
                    """)
                
                if row['Provider'] == "TII":
                    st.markdown("""
                    **Via Hugging Face (TII Models):**
                    1. Search "Falcon" on [Hugging Face](https://huggingface.co/models)
                    2. Commercial use permitted under Apache 2.0
                    3. Deploy with vLLM, LM Studio, or Ollama
                    """)
                
                if row['Provider'] == "EleutherAI":
                    st.markdown("""
                    **Via Hugging Face (EleutherAI):**
                    1. Find EleutherAI models on Hugging Face
                    2. Download weights or use transformers library
                    3. MIT license - fully open source
                    """)
                
                if "GitHub" in row['Access Method']:
                    st.markdown("""
                    **Via GitHub:**
                    1. Clone repository
                    2. Install dependencies
                    3. Download model weights
                    """)
                
                # Common deployment platforms
                st.markdown("#### 🚀 Recommended Deployment Platforms")
                
                deployment_info = {
                    "Local": "Run on your own hardware using Ollama, LM Studio, or vLLM",
                    "Hugging Face Spaces": "Free GPU inference at huggingface.co/spaces",
                    "Replicate": "Pay-per-use API service with automatic scaling",
                    "Together AI": "Shared GPU cluster for cost-effective inference",
                    "Modal": "Run serverless GPU workloads",
                    "RunwayML": "Web-based interface for model inference"
                }
                
                for platform, desc in deployment_info.items():
                    st.write(f"• **{platform}**: {desc}")
                
                # Cost comparison
                st.markdown("#### 💰 Cost Information")
                st.info("""
                **Open Source Advantages:**
                - ✅ No API costs for local deployment
                - ✅ No rate limits
                - ✅ Full model customization
                - ✅ Data privacy (runs locally)
                - ⚠️ Requires GPU hardware investment
                """)
    else:
        st.warning("No open source models in current filters. Adjust filters to see open source options.")
    
    st.markdown("---")
    
    # Open Source Model Comparison Table
    st.subheader("Open Source vs Proprietary Comparison")
    
    comparison_data = {
        "Aspect": [
            "Cost (Inference)",
            "Model Customization",
            "Data Privacy",
            "Rate Limits",
            "Infrastructure",
            "Support",
            "Commercial Use",
            "Latency Control"
        ],
        "Open Source": [
            "Free (local) or pay-per-use",
            "Full - retrain/fine-tune",
            "Complete - runs locally",
            "None - self-hosted",
            "Own GPU hardware needed",
            "Community support",
            "Usually allowed (check license)",
            "Full control"
        ],
        "Proprietary (OpenAI)": [
            "Pay-per-token usage",
            "Limited to prompting",
            "Data sent to servers",
            "API rate limits apply",
            "OpenAI's infrastructure",
            "Professional support available",
            "Allowed per terms",
            "OpenAI controls"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><small>GPT Model Comparison Dashboard | Data based on official specifications (as of December 2024)</small></p>
</div>
""", unsafe_allow_html=True)
