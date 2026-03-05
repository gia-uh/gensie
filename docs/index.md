# Welcome to GenSIE 2026

**GenSIE (General-purpose Schema-guided Information Extraction)** is a shared task at [IberLEF 2026](https://sites.google.com/view/iberlef-2026) focusing on the ability of systems to extract nested, structured information (JSON) from general-domain Spanish texts.

> [Read the full task description](./gensie.pdf), including score metrics and detailed constraints.

The task challenges participants to use **Small Language Models (SLMs)** and inference-time techniques to handle **Zero-Shot Schemas**—where the extraction target is defined dynamically at runtime.


<div class="grid cards">
  <ul>
    <li>
      <p>
        <span class="twemoji">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h-2V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5A2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5a2.5 2.5 0 0 0 2.5 2.5a2.5 2.5 0 0 0 2.5-2.5a2.5 2.5 0 0 0-2.5-2.5M12 15c-2.7 0-5.8 1.29-6 4v1h12v-1c-.2-2.71-3.3-4-6-4m0-7a3 3 0 0 1 3 3v2h-6V8a3 3 0 0 1 3-3Z"/></svg>
        </span>
        <strong>Zero-Shot Schema</strong>
      </p>
      <hr>
      <p>Extract data using schemas seen only at inference time. No fixed ontology.</p>
    </li>
    <li>
      <p>
        <span class="twemoji">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M12 3 2 8v2h20V8l-10-5m0 10c-4.42 0-8 3.58-8 8h16c0-4.42-3.58-8-8-8Z"/></svg>
        </span>
        <strong>General Domain</strong>
      </p>
      <hr>
      <p>From legal contracts to medical reports and news.</p>
    </li>
    <li>
      <p>
        <span class="twemoji">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M7 2v11h3v9l7-12h-4l4-8H7Z"/></svg>
        </span>
        <strong>Inference-Time Focus</strong>
      </p>
      <hr>
      <p>Focus on prompting, RAG, and constrained decoding. No massive fine-tuning.</p>
    </li>
    <li>
      <p>
        <span class="twemoji">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M5 3h2v2H5v14h2v2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2m14 0h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-2v-2h2V5h-2V3m-9 10H8v2h2v-2m6 0h-2v2h2v-2m-3 0h-2v2h2v-2Z"/></svg>
        </span>
        <strong>Structured Output</strong>
      </p>
      <hr>
      <p>Strict adherence to JSON Schema and complex semantic constraints.</p>
    </li>
  </ul>
</div>


## Schedule

| Date | Event |
| :--- | :--- |
| ⚠️ **March 09, 2026** | 🚀 **Starter Kit Release** (Baselines, Docker templates, 30-example Dev Set) |
| **April 01, 2026** | 📂 **Full Development Set** (Remaining 170 examples) |
| **May 08, 2026** | 🛑 **Submission Deadline** (Docker containers) |
| **May 09, 2026** | 🔓 **Test Set Release** (For local error analysis) |
| **May 09–30, 2026** | ⚙️ **Evaluation Period** (Hosted execution) |
| **May 31, 2026** | 🏆 **Results Announcement** |
| **June 07, 2026** | 📝 **Paper Submission Deadline** |
| **Sept 22, 2026** | 🎤 **IberLEF Workshop** (León, Spain) |

## News & Updates

* **Jan 26, 2026:** Website launched.
* **March 01, 2026:** We've had some delays with the preparation of the _starter-kit_ which forced to push the date back to **March 09** at the latest.

## Motivation

The rise of **Agentic Workflows** has created a massive demand for systems that can communicate via structured protocols. To identify user intent, invoke external tools, or exchange information, an AI must output rigid, error-free structured data.

While massive proprietary models (like GPT-5) solve this through scale, **GenSIE** targets the innovation gap in **Small Language Models (<14B)**. We aim to prove that with clever engineering (Chain-of-Thought, ReAct, Constrained Decoding), commodity hardware can perform complex structured extraction reliably.

## Organizing Committee

The GenSIE task is organized by a consortium between the **Research Group on Artificial Intelligence and Data Science (GIA-UH)** at the University of Havana and the **Research Group in Natural Language Processing and Information Systems (GPLSI)** at the University of Alicante.

This team brings together expertise in both Computer Science (Generative AI, Large Language Models) and Linguistics (Corpus Annotation, Semantic Evaluation).

### Members

| Name | Affiliation | Role |
| :--- | :--- | :--- |
| **Yudivian Almeida Cruz** | University of Havana | PhD, Professor |
| **Suilan Estévez Velarde** | University of Havana | PhD, Professor |
| **Alejandro Piad Morffis** | University of Havana | PhD, Professor |
| **Isabel Espinosa Zaragoza** | University of Alicante | PhD, Assistant Professor |
| **María Miró Maestre** | University of Alicante | PhD, Postdoc Researcher |
| **Lucía Sevilla Requena** | University of Alicante | PhD Student, Assoc. Prof. |
| **Alba Pérez Montero** | University of Alicante | PhD Student |
| **Ernesto Estevanell Valladares** | University of Havana | PhD Student |

### Contact

For questions regarding the task, dataset, or evaluation, please contact the corresponding author, [**Alejandro Piad Morffis**](mailto:apiad@matcom.uh.cu).
