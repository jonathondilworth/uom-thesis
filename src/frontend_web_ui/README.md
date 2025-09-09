# Knowledge Retrieval Web Demo

This repository contains the web demo UI for the knowledge retrieval thesis. Features include ontology-aware retrival and a basic RAG pipeline.

## Usage

1. **Install dependencies:**  
   
```bash
npm install
```

```bash
npm run dev
```

2. **Run Backend**

*(from project root, run)*

```bash
make webapp
```

...or (run from project root with appropriate deps installed):

```bash
python src/thesis/app/serve_single_host.py
```

3. Enjoy!