# MEMORY.md

## Project: AI Vision App
- **Goal:** Build an object recognition app using the latest Ultralytics YOLO model, targeting the iPhone camera.
- **Current Hypothesis:** Web app with camera access (rather than native iOS).
- **Architecture:** Option A - Client-Side Inference (ONNX/TFJS in browser).
- **Core Workflow:**
  1. GPS Reverse Geocode (Nominatim) for Address.
  2. Input Room ID.
  3. Bounding Box "Click-to-Catalog" (prevents double counting).
  4. Client-side PDF generation (jsPDF) with item crops and generated descriptions.
- **Team Roles:** 
  - Lucas: Project Owner (PO)
  - Alastair: Team member / Developer
- **Tech Stack (Initial):** Ultralytics YOLO (latest), Web stack (HTML/JS/Camera API).