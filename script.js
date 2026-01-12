// DOM Elements
const micButton = document.getElementById('micButton');
const statusDisplay = document.getElementById('statusDisplay');
const timer = document.getElementById('timer');
const konkaniText = document.getElementById('konkaniText');
const englishText = document.getElementById('englishText');
const emotionBadge = document.getElementById('emotionBadge');
const emotionConfidence = document.getElementById('emotionConfidence');
const clearBtn = document.getElementById('clearBtn');
const errorMessage = document.getElementById('errorMessage');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const visualizerBars = document.querySelectorAll('.visualizer-bar');

// State
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = null;
let timerInterval = null;
let websocket = null;

// Charts
let emotionPieChart = null;
let emotionTimeChart = null;
let confidenceChart = null;
let intensityChart = null;

// Initialize Charts
function initializeCharts() {
    const pieCtx = document.getElementById('emotionPieChart').getContext('2d');
    emotionPieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Surprise'],
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    '#d4edda',
                    '#f8d7da',
                    '#f5c6cb',
                    '#d1ecf1',
                    '#fff3cd',
                    '#cfe2ff'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    const timeCtx = document.getElementById('emotionTimeChart').getContext('2d');
    emotionTimeChart = new Chart(timeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Emotion Score',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });

    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'bar',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Surprise'],
            datasets: [{
                label: 'Confidence %',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

    const intensityCtx = document.getElementById('intensityChart').getContext('2d');
    intensityChart = new Chart(intensityCtx, {
        type: 'radar',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Surprise'],
            datasets: [{
                label: 'Intensity',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: '#667eea',
                pointBackgroundColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Connect to WebSocket
function connectWebSocket() {
    // TODO: Replace with your actual WebSocket endpoint
    const wsUrl = 'ws://localhost:8000/ws';
    
    try {
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            statusIndicator.classList.remove('disconnected');
            statusText.textContent = 'Connected';
            micButton.disabled = false;
            hideError();
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerResponse(data);
        };

        websocket.onerror = (error) => {
            showError('WebSocket error. Please check backend connection.');
            console.error('WebSocket error:', error);
        };

        websocket.onclose = () => {
            statusIndicator.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
            micButton.disabled = true;
            showError('Connection lost. Please refresh the page.');
        };
    } catch (error) {
        showError('Failed to connect to server');
        console.error('Connection error:', error);
    }
}

// Handle server response
function handleServerResponse(data) {
    // Update transcription
    if (data.konkani) {
        konkaniText.textContent = data.konkani;
    }

    if (data.english) {
        englishText.textContent = data.english;
    }

    // Update emotion
    if (data.emotion) {
        updateEmotion(data.emotion);
    }

    // Update charts
    if (data.emotionData) {
        updateCharts(data.emotionData);
    }
}

// Update emotion display
function updateEmotion(emotionData) {
    const emotion = emotionData.label || 'neutral';
    const confidence = emotionData.confidence || 0;

    emotionBadge.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    emotionBadge.className = 'emotion-badge emotion-' + emotion.toLowerCase();
    emotionConfidence.textContent = `${(confidence * 100).toFixed(1)}% confident`;
}

// Update all charts
function updateCharts(emotionData) {
    // Update pie chart (distribution)
    if (emotionData.distribution) {
        emotionPieChart.data.datasets[0].data = emotionData.distribution;
        emotionPieChart.update();
    }

    // Update time series
    if (emotionData.timestamp && emotionData.score !== undefined) {
        emotionTimeChart.data.labels.push(emotionData.timestamp);
        emotionTimeChart.data.datasets[0].data.push(emotionData.score);
        
        // Keep only last 20 points
        if (emotionTimeChart.data.labels.length > 20) {
            emotionTimeChart.data.labels.shift();
            emotionTimeChart.data.datasets[0].data.shift();
        }
        emotionTimeChart.update();
    }

    // Update confidence chart
    if (emotionData.confidences) {
        confidenceChart.data.datasets[0].data = emotionData.confidences.map(c => c * 100);
        confidenceChart.update();
    }

    // Update intensity radar
    if (emotionData.intensities) {
        intensityChart.data.datasets[0].data = emotionData.intensities;
        intensityChart.update();
    }
}

// Start/Stop Recording
micButton.addEventListener('click', async () => {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
            
            // Send audio chunk to server via WebSocket
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(event.data);
            }
        };

        mediaRecorder.start(100); // Send data every 100ms
        isRecording = true;
        micButton.classList.add('recording');
        statusDisplay.textContent = 'Recording...';
        statusDisplay.classList.add('recording');
        
        recordingStartTime = Date.now();
        startTimer();
        animateVisualizer();

    } catch (error) {
        showError('Microphone access denied. Please allow microphone permissions.');
        console.error('Error accessing microphone:', error);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }

    isRecording = false;
    micButton.classList.remove('recording');
    statusDisplay.textContent = 'Click microphone to start';
    statusDisplay.classList.remove('recording');
    
    stopTimer();
    stopVisualizer();
}

// Timer functions
function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - recordingStartTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        timer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// Visualizer animation
function animateVisualizer() {
    if (!isRecording) return;

    visualizerBars.forEach(bar => {
        const height = Math.random() * 60 + 10;
        bar.style.height = height + 'px';
    });

    requestAnimationFrame(animateVisualizer);
}

function stopVisualizer() {
    visualizerBars.forEach(bar => {
        bar.style.height = '20px';
    });
}

// Clear all data
clearBtn.addEventListener('click', () => {
    konkaniText.textContent = 'Your speech will appear here...';
    englishText.textContent = 'Translation will appear here...';
    emotionBadge.textContent = 'Waiting...';
    emotionBadge.className = 'emotion-badge emotion-default';
    emotionConfidence.textContent = '';
    timer.textContent = '00:00';
    
    // Reset charts
    emotionPieChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
    emotionPieChart.update();
    
    emotionTimeChart.data.labels = [];
    emotionTimeChart.data.datasets[0].data = [];
    emotionTimeChart.update();
    
    confidenceChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
    confidenceChart.update();
    
    intensityChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
    intensityChart.update();
});

// Error handling
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

// Initialize on page load
window.addEventListener('load', () => {
    initializeCharts();
    connectWebSocket();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isRecording) {
        stopRecording();
    }
    if (websocket) {
        websocket.close();
    }
});

// ============================================================
// MOCK MODE FOR TESTING WITHOUT BACKEND
// Uncomment the section below to test with mock data
// Comment out the connectWebSocket() call above
// ============================================================

/*
function mockWebSocket() {
    statusIndicator.classList.remove('disconnected');
    statusText.textContent = 'Connected (Mock Mode)';
    micButton.disabled = false;

    // Simulate receiving data every 2 seconds during recording
    setInterval(() => {
        if (!isRecording) return;

        const emotions = ["happy", "neutral", "sad", "angry", "fear", "surprise"];
        const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];

        const mockData = {
            konkani: "हाँव बरें आसा",
            english: "I am fine",
            emotion: {
                label: randomEmotion,
                confidence: 0.7 + Math.random() * 0.3
            },
            emotionData: {
                distribution: [30, 15, 10, 35, 5, 5],
                timestamp: timer.textContent,
                score: Math.random(),
                confidences: [0.3, 0.15, 0.1, 0.35, 0.05, 0.05],
                intensities: [
                    Math.random(),
                    Math.random(),
                    Math.random(),
                    Math.random(),
                    Math.random(),
                    Math.random()
                ]
            }
        };

        handleServerResponse(mockData);
    }, 2000);
}

// Replace connectWebSocket() in window.addEventListener('load') with:
// mockWebSocket();
*/