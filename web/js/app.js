// Splash Screen Logic
document.addEventListener("DOMContentLoaded", () => {
    const splash = document.getElementById('splash-screen');
    const setup = document.getElementById('setup-screen');
    setTimeout(() => {
        splash.style.opacity = '0';
        setTimeout(() => {
            splash.style.display = 'none';
            setup.style.display = 'flex';
        }, 1000);
    }, 2500); // Wait 2.5 seconds before fading out
});

const video = document.getElementById('camera-feed');
const canvas = document.getElementById('bounding-boxes');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status-text');
const controls = document.getElementById('controls');
const confSlider = document.getElementById('conf-slider');
const confValue = document.getElementById('conf-value');

// State
let session;
let confidenceThreshold = 0.4;
let catalog = [];
let address = '';
let roomId = '';
let currentDrawnBoxes = [];

const modelInputShape = [1, 3, 640, 640];
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

confSlider.addEventListener('input', (e) => {
    confValue.innerText = e.target.value + '%';
    confidenceThreshold = parseInt(e.target.value) / 100.0;
});

// --- SETUP PHASE (GPS & ROOM) ---
const setupScreen = document.getElementById('setup-screen');
const appContainer = document.getElementById('app-container');
const addrInput = document.getElementById('address-input');
const roomInput = document.getElementById('room-input');
const beginBtn = document.getElementById('begin-btn');
const locStatus = document.getElementById('location-status');

function checkSetupForm() {
    if (addrInput.value.trim() !== '' && roomInput.value.trim() !== '') {
        beginBtn.disabled = false;
    } else {
        beginBtn.disabled = true;
    }
}
addrInput.addEventListener('input', checkSetupForm);
roomInput.addEventListener('input', checkSetupForm);

navigator.geolocation.getCurrentPosition(async (pos) => {
    try {
        locStatus.innerText = "Looking up address...";
        const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${pos.coords.latitude}&lon=${pos.coords.longitude}`);
        const data = await res.json();
        if (data && data.display_name) {
            addrInput.value = data.display_name;
            locStatus.innerText = "GPS Location Acquired.";
            checkSetupForm();
        } else {
            locStatus.innerText = "Could not resolve address. Please type it.";
        }
    } catch (e) {
        locStatus.innerText = "GPS error. Please type your address.";
    }
}, () => {
    locStatus.innerText = "GPS permission denied. Please type your address.";
});

beginBtn.addEventListener('click', () => {
    address = addrInput.value.trim();
    roomId = roomInput.value.trim();
    setupScreen.style.display = 'none';
    appContainer.style.display = 'flex';
    initCamera();
});

// --- CAMERA & MODEL INFERENCE ---
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });
        video.srcObject = stream;
        video.onloadedmetadata = async () => {
            video.play();
            resizeCanvas();
            await loadModel();
            runInference();
        };
    } catch (error) {
        console.error("Error accessing camera:", error);
        statusText.innerText = "Error: Camera access denied.";
    }
}

function resizeCanvas() {
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
}
window.addEventListener('resize', resizeCanvas);

async function loadModel() {
    statusText.innerText = "Loading AI Model (~10MB)...";
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    session = await ort.InferenceSession.create('model/yolo11n.onnx', { executionProviders: ['wasm'] });
    statusText.innerText = "Model loaded. Ready!";
    statusText.style.display = "none";
    controls.style.display = "block";
}

function preprocessImage() {
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = 640; offscreenCanvas.height = 640;
    const offCtx = offscreenCanvas.getContext('2d');
    
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - minDim) / 2;
    const startY = (video.videoHeight - minDim) / 2;
    offCtx.drawImage(video, startX, startY, minDim, minDim, 0, 0, 640, 640);

    const imgData = offCtx.getImageData(0, 0, 640, 640);
    const data = imgData.data;
    const float32Data = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        float32Data[i] = data[i * 4] / 255.0; // R
        float32Data[i + 640 * 640] = data[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * 640 * 640] = data[i * 4 + 2] / 255.0; // B
    }
    return new ort.Tensor('float32', float32Data, modelInputShape);
}

function calculateIoU(box1, box2) {
    const [x1, y1, w1, h1] = box1; const [x2, y2, w2, h2] = box2;
    const x_left = Math.max(x1, x2); const y_top = Math.max(y1, y2);
    const x_right = Math.min(x1 + w1, x2 + w2); const y_bottom = Math.min(y1 + h1, y2 + h2);
    if (x_right < x_left || y_bottom < y_top) return 0.0;
    const intersection_area = (x_right - x_left) * (y_bottom - y_top);
    return intersection_area / (w1 * h1 + w2 * h2 - intersection_area);
}

function nonMaxSuppression(detections, iouThreshold = 0.45) {
    const byClass = {};
    for (const det of detections) {
        if (!byClass[det.classId]) byClass[det.classId] = [];
        byClass[det.classId].push(det);
    }
    const finalDetections = [];
    for (const classId in byClass) {
        let classDets = byClass[classId].sort((a, b) => b.prob - a.prob);
        while (classDets.length > 0) {
            const best = classDets[0];
            finalDetections.push(best);
            classDets = classDets.slice(1).filter(det => calculateIoU(best.box, det.box) < iouThreshold);
        }
    }
    return finalDetections;
}

async function runInference() {
    if (!session) return;
    const tensor = preprocessImage();
    const results = await session.run({ images: tensor });
    const output = results[session.outputNames[0]].data;
    
    const numBoxes = 8400; const numClasses = 80;
    let detections = [];

    for (let index = 0; index < numBoxes; index++) {
        let maxClassProb = 0; let classId = -1;
        for (let col = 0; col < numClasses; col++) {
            const prob = output[(4 + col) * numBoxes + index];
            if (prob > maxClassProb) { maxClassProb = prob; classId = col; }
        }
        if (maxClassProb > confidenceThreshold) {
            const x = output[0 * numBoxes + index]; const y = output[1 * numBoxes + index];
            const w = output[2 * numBoxes + index]; const h = output[3 * numBoxes + index];
            detections.push({ box: [x - w/2, y - h/2, w, h], classId, prob: maxClassProb });
        }
    }

    drawBoxes(nonMaxSuppression(detections));
    requestAnimationFrame(runInference);
}

function drawBoxes(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const minDim = Math.min(canvas.width, canvas.height);
    const scale = minDim / 640;
    const offsetX = (canvas.width - minDim) / 2;
    const offsetY = (canvas.height - minDim) / 2;

    currentDrawnBoxes = [];

    detections.forEach(det => {
        const [x, y, w, h] = det.box;
        const scaledX = (x * scale) + offsetX;
        const scaledY = (y * scale) + offsetY;
        const scaledW = w * scale;
        const scaledH = h * scale;

        currentDrawnBoxes.push({
            scaledX, scaledY, scaledW, scaledH,
            classId: det.classId, prob: det.prob, rawBox: det.box
        });

        const hue = (det.classId * 360) / 80;
        const color = `hsl(${hue}, 100%, 50%)`;

        ctx.strokeStyle = color; ctx.lineWidth = 4;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        
        ctx.fillStyle = color;
        const text = `${COCO_CLASSES[det.classId]} (Tap to Catalog)`;
        ctx.font = '16px Arial';
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(scaledX, scaledY - 24, textWidth + 10, 24);
        
        ctx.fillStyle = '#000000';
        ctx.fillText(text, scaledX + 5, scaledY - 7);
    });
}

// --- CATALOG INTERACTION ---
function captureCrop(box) {
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - minDim) / 2;
    const startY = (video.videoHeight - minDim) / 2;
    const scale = minDim / 640;
    
    // Map from 640x640 space back to the native video pixel space
    const vX = startX + (box[0] * scale);
    const vY = startY + (box[1] * scale);
    const vW = box[2] * scale;
    const vH = box[3] * scale;

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = vW; cropCanvas.height = vH;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(video, vX, vY, vW, vH, 0, 0, vW, vH);
    return cropCanvas.toDataURL('image/jpeg', 0.8);
}

canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Find smallest box clicked (handles overlaps)
    let clickedBox = null; let minArea = Infinity;
    for (const box of currentDrawnBoxes) {
        if (clickX >= box.scaledX && clickX <= box.scaledX + box.scaledW &&
            clickY >= box.scaledY && clickY <= box.scaledY + box.scaledH) {
            const area = box.scaledW * box.scaledH;
            if (area < minArea) { minArea = area; clickedBox = box; }
        }
    }
    
    if (clickedBox) {
        const name = COCO_CLASSES[clickedBox.classId];
        const imgData = captureCrop(clickedBox.rawBox);
        const desc = `Identified as a ${name} with ${Math.round(clickedBox.prob * 100)}% confidence at ${address}.`;
        
        // Generate a unique 8-character inventory ID
        const invId = 'LV-' + Math.random().toString(36).substr(2, 6).toUpperCase();
        
        catalog.push({ id: invId, name, desc, imgData });
        document.getElementById('catalog-count').innerText = `Items: ${catalog.length}`;
        
        // Show flash message
        const flash = document.getElementById('flash-msg');
        flash.style.display = 'block';
        setTimeout(() => { flash.style.display = 'none'; }, 1000);
    }
});

// --- PDF GENERATION ---
document.getElementById('finish-btn').addEventListener('click', () => {
    if (catalog.length === 0) {
        alert("Catalog is empty!"); return;
    }
    
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    doc.setFontSize(22); doc.text(`Room Inventory Catalog`, 10, 20);
    doc.setFontSize(12); doc.text(`Address: ${address}`, 10, 30);
    doc.text(`Room ID: ${roomId}`, 10, 38);
    doc.text(`Date Cataloged: ${new Date().toLocaleString()}`, 10, 46);
    
    let yOffset = 60;
    catalog.forEach((item, index) => {
        if (yOffset > 240) { doc.addPage(); yOffset = 20; }
        
        doc.setFontSize(14); doc.text(`${index + 1}. ${item.name.toUpperCase()} (ID: ${item.id})`, 10, yOffset);
        doc.setFontSize(10); doc.text(item.desc, 10, yOffset + 6);
        
        // Add image (preserving aspect ratio)
        const imgProps = doc.getImageProperties(item.imgData);
        const pdfWidth = 50;
        const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
        
        doc.addImage(item.imgData, 'JPEG', 10, yOffset + 10, pdfWidth, pdfHeight);
        
        yOffset += pdfHeight + 25; // Advance cursor
    });
    
    // Format filename safely
    const safeAddr = address.substring(0, 15).replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const safeRoom = roomId.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    doc.save(`${safeAddr}_${safeRoom}.pdf`);
});