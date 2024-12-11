# Automated Incident Response: Leveraging LLMs for Rapid Post-Attack Analysis and Reporting

An advanced framework for automated incident response, integrating Machine Learning-based classifiers with the reasoning capabilities of Large Language Models (LLMs). This system enables rapid post-attack analysis, efficient threat correlation, and comprehensive reporting to streamline cybersecurity workflows.

---

## 🌟 Key Features

- **Multi-Layered Architecture**: Specialized analysis and orchestration layers for streamlined log processing and analysis.
- **Cross-Log Correlation**: Comprehensive insights via orchestration of data from diverse log sources.
- **Automated Reporting**: Structured, detailed incident reports with attack timelines, IoCs, and mitigation strategies.
- **Feedback-Driven Learning**: Human analyst feedback loop for continuous model improvement.
- **Seamless Integration**: API support for SIEM and SOAR tools, ensuring compatibility with existing security infrastructure.

---

## 🛠️ System Architecture

<img width="1038" alt="diagram" src="https://github.com/user-attachments/assets/145aff43-0735-42ba-aa5a-dbd4f6b0785c">

1. **Log Ingestion**: Prepares and standardizes logs from Suricata, Wazuh, SMTP, and network traffic.
2. **Classification**: Machine Learning-based classifiers for anomaly detection across different log types.
3. **Orchestration**: LLM-powered correlation and narrative generation for enhanced incident understanding.
4. **Report Generation**: Automated creation of multi-format reports (HTML, PDF).
5. **Human Feedback**: Analyst reviews refine the system's accuracy and adaptability.
---

## 🗂️ Code Structure

```
├── main.py                      # Entry point of the application
├── llm/
│   ├── llm_client.py            # LLM client interface
│   ├── llm_server.py            # LLM server setup
├── smtp_classifier/             # SMTP log classification
├── surikata_classifier/         # Suricata log analysis
├── wazuh_classifier/            # Wazuh log processing
└── docs/                        # Documentation and related files
```

---

## 📋 Installation

### Prerequisites
- Python 3.8+
- pipenv or virtualenv
- Docker (optional for easier deployment)
- GPUs for model training and inference (optional)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/siddhant-192/Automated-Incidence-Response.git
   cd Automated-Incidence-Response
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

4. Run the application:
   ```bash
   python main.py
   ```

---
## 📖 Usage

1.	Upload Logs: Use the CLI or API to input logs.
2.	Classify Data: ML models parse and flag anomalies.
3.	Correlate and Analyze: Orchestrator LLM consolidates findings.
4.	Review Reports: Analysts review and act on structured outputs.

---
## 💬 Feedback and Contributions

We welcome contributions! To get involved:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Open a pull request with your changes.

For queries or suggestions, contact us at: [siddhantmantri328@gmail.com](mailto:siddhantmantri328@gmail.com)

---
## 👨‍💻 Contributors

- Siddhant H Mantri
- Veer Mehta
- Aryamann Khare
- **Mentor**: Dr. Pintu R Shah, SVKM’s NMIMS - MPSTME

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## 📬 Contact

For any queries or collaboration opportunities, please contact:
- **Email**: siddhantmantri328@gmail.com
- **GitHub Issues**: [Issue Tracker](https://github.com/siddhant-192/Automated-Incidence-Response/issues)
