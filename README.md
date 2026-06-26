# Multi-Disease-Diagnosis-Platform
An end-to-end medical suite featuring different, significant medical diseases diagnosis.
# Multi-Disease AI Diagnostics Suite 🩺

An enterprise-grade, full-stack predictive healthcare analytics platform. The application implements a highly scalable, single-endpoint dynamic routing backend designed to orchestrate, transform, and evaluate multi-parametric patient clinical biomarkers across 6 medical units using trained machine learning models.

---

## 🚀 Core Architectural Enhancements
Hiring managers can review the major full-stack and machine learning design patterns applied to this codebase below:

* **Dynamic Polymorphic View Pattern:** Instead of hardcoding 6 separate and redundant HTML pages, the system routes all diagnostic entries through a single, highly reusable frontend view template (`prediction.html`). An embedded script switches the visible CSS grid input blocks instantly based on backend metadata.
* **Feature Ingestion Data Padding Layer:** Real-world machine learning models require highly explicit matrix dimensional shapes. The Flask backend uses an array padding architecture via **NumPy** to dynamically intercept incoming data records and append zero-value vectors (`0.0`) up to strict dimensional array shapes (e.g., expanding 5 custom lab attributes up to a 25-feature matrix structure required by the trained classifier).
* **Flat Core Workspace Hierarchy:** Reconfigured the Flask view rendering framework by overriding standard path variables (`template_folder='.'`). This eliminates nested template folder overhead and anchors the server files, markup templates, and compiled binary models (`.pkl`) side-by-side in a clean workspace scope.

---

## 📁 System Blueprint & Repository Structure

```text
Medical Diagnosis/
├── app.py                      # Core Routing Engine & Feature Transformation Pipeline
├── login.html                  # Secure Authentication Access Gateway Portal
├── register.html               # Security Credentials Provisioning Layout
├── dashboard.html              # Multi-Department Diagnostic Selection Hub
├── prediction.html             # Combined Adaptive Matrix Entry Panel
├── result.html                 # Confirmed Diagnostic Outcome Analytics Display
├── CKD RandomForest.pkl        # Live Nephrology Production Model (25-Feature Expectation)
└── Liver RandomForest.pkl      # Live Hepatology Production Model (36-Feature Expectation)
