from setuptools import setup, find_packages

setup(
    name="ai_strategy_lab",
    version="0.1.0",
    description="AI Strategy Lab for crypto trading with WebSocket-based real-time data streaming",
    author="The Bulls Team",
    author_email="team@thebulls.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
        install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "pydantic>=2.11.0",
        "starlette>=0.46.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.28.0",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-community>=0.3.0",
        "websockets>=15.0.0",
        "pandas>=2.1.3",
        "numpy>=1.26.2",
        "gymnasium>=1.0.0",
        "python-telegram-bot==22.1"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "llm": [
            "sentence-transformers>=2.2.2",
            "chromadb>=0.4.18",
            "faiss-cpu>=1.7.4",
            "openai>=1.3.5",
        ],
        "agents": [
            "pyautogen>=0.2.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-strategy-lab=app.main:run_app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
