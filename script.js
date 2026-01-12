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
const consoleOutput = document.getElementById('consoleOutput');
const clearConsoleBtn = document.getElementById('clearConsoleBtn');

// State
let isRecording = false;
let mediaRecorder = null;
let recordingStartTime = null;
let timerInterval = null;
let websocket = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// Charts
let emotionPieChart = null;
let emotionTimeChart = null;
let confidenceChart = null;
let intensityChart = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

// Initialize Charts (DISABLED - performance optimization)
function initializeCharts() {
    // Charts disabled to reduce frontend load
    console.log('Charts disabled for performance');
    return;
}

// ============================================================================
// CONSOLE LOGGING
// ============================================================================

function logToConsole(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function clearConsole() {
    consoleOutput.innerHTML = '<div class="console-line">Console cleared.</div>';
}

// ============================================================================
// WEBSOCKET CONNECTION
// ============================================================================

function connectWebSocket() {
    // Always connect to localhost:8000 for the WebSocket server
    const wsUrl = 'ws://localhost:8000/ws';
    
    logToConsole('Connecting to server...', 'info');
    statusText.textContent = 'Connecting...';
    
    try {
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            statusIndicator.classList.remove('disconnected');
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Connected';
            micButton.disabled = false;
            hideError();
            reconnectAttempts = 0;
            logToConsole('✓ Connected to WebSocket server', 'success');
        };

        websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerResponse(data);
                logToConsole('✓ Received response from server', 'success');
            } catch (error) {
                logToConsole(`✗ Error parsing message: ${error.message}`, 'error');
            }
        };

        websocket.onerror = (error) => {
            showError('WebSocket error. Please check backend connection.');
            console.error('WebSocket error:', error);
            logToConsole('✗ WebSocket error occurred', 'error');
        };

        websocket.onclose = () => {
            statusIndicator.classList.remove('connected');
            statusIndicator.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
            micButton.disabled = true;
            
            // Stop recording if active
            if (isRecording) {
                stopRecording();
            }
            
            logToConsole('✗ Connection closed', 'error');
            
            // Attempt reconnection
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                reconnectAttempts++;
                const delay = Math.min(1000 * reconnectAttempts, 5000);
                logToConsole(`Reconnecting in ${delay/1000}s (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`, 'warning');
                setTimeout(connectWebSocket, delay);
            } else {
                showError('Connection lost. Please refresh the page.');
            }
        };
    } catch (error) {
        showError('Failed to connect to server');
        console.error('Connection error:', error);
        logToConsole(`✗ Connection failed: ${error.message}`, 'error');
    }
}

// ============================================================================
// SERVER RESPONSE HANDLER
// ============================================================================

function handleServerResponse(data) {
    console.log('handleServerResponse called with:', data);
    
    // Handle errors
    if (data.error) {
        logToConsole(`✗ Server error: ${data.error}`, 'error');
        return;
    }
    
    // Handle no speech detected
    if (data.status === 'no_speech') {
        logToConsole('No speech detected in audio chunk', 'warning');
        return;
    }
    
    // Update transcription
    if (data.konkani) {
        konkaniText.textContent = data.konkani;
        logToConsole(`Konkani: ${data.konkani}`, 'success');
    }

    if (data.english) {
        englishText.textContent = data.english;
        logToConsole(`English: ${data.english}`, 'success');
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

// ============================================================================
// EMOTION DISPLAY
// ============================================================================

function updateEmotion(emotionData) {
    const emotion = emotionData.label || 'neutral';
    const confidence = emotionData.confidence || 0;

    emotionBadge.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    emotionBadge.className = 'emotion-badge emotion-' + emotion.toLowerCase();
    emotionConfidence.textContent = `${(confidence * 100).toFixed(1)}% confident`;
    
    logToConsole(`Emotion: ${emotion} (${(confidence * 100).toFixed(1)}%)`, 'info');
}

// ============================================================================
// CHARTS UPDATE
// ============================================================================

function updateCharts(emotionData) {
    // Update pie chart (distribution)
    if (emotionData.distribution) {
        emotionPieChart.data.datasets[0].data = emotionData.distribution;
        emotionPieChart.update();
    }

    // Update time series
    if (emotionData.score !== undefined) {
        const timestamp = new Date().toLocaleTimeString();
        emotionTimeChart.data.labels.push(timestamp);
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

// ============================================================================
// RECORDING CONTROLS
// ============================================================================

micButton.addEventListener('click', async () => {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording() {
    try {
        logToConsole('🎤 Requesting microphone access...', 'info');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: { ideal: 16000 },
                echoCancellation: true,
                noiseSuppression: false,  // Disable to preserve Konkani speech
                autoGainControl: true,
                latency: 0
            } 
        });
        
        // Log the actual audio track settings
        const audioTrack = stream.getAudioTracks()[0];
        const settings = audioTrack.getSettings();
        logToConsole(`✓ Microphone: ${audioTrack.label}`, 'success');
        logToConsole(`  Sample Rate: ${settings.sampleRate}Hz, Channels: ${settings.channelCount}`, 'info');
        
        // Use WebM format for better compatibility
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
            ? 'audio/webm;codecs=opus' 
            : 'audio/webm';
            
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        
        logToConsole(`✓ Recording format: ${mimeType}`, 'success');

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                // Send audio chunk to server
                websocket.send(event.data);
                logToConsole(`→ Sent audio chunk (${event.data.size} bytes)`, 'info');
            }
        };

        mediaRecorder.onerror = (event) => {
            logToConsole(`✗ MediaRecorder error: ${event.error}`, 'error');
            stopRecording();
        };

        // Send data every 2 seconds for better speech capture
        mediaRecorder.start(2000);
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
        logToConsole(`✗ Microphone error: ${error.message}`, 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }

    logToConsole('⏹ Recording stopped', 'warning');
    isRecording = false;
    micButton.classList.remove('recording');
    statusDisplay.textContent = 'Click microphone to start';
    statusDisplay.classList.remove('recording');
    
    stopTimer();
    stopVisualizer();
}

// ============================================================================
// TIMER
// ============================================================================

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

// ============================================================================
// VISUALIZER
// ============================================================================

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

// ============================================================================
// CLEAR FUNCTIONS
// ============================================================================

clearConsoleBtn.addEventListener('click', () => {
    clearConsole();
});

clearBtn.addEventListener('click', () => {
    logToConsole('🗑️ Clearing all data...', 'warning');
    konkaniText.textContent = 'Your speech will appear here...';
    englishText.textContent = 'Translation will appear here...';
    emotionBadge.textContent = 'Waiting...';
    emotionBadge.className = 'emotion-badge emotion-default';
    emotionConfidence.textContent = '';
    timer.textContent = '00:00';
    
    // Reset charts
    if (emotionPieChart) {
        emotionPieChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
        emotionPieChart.update();
    }
    
    if (emotionTimeChart) {
        emotionTimeChart.data.labels = [];
        emotionTimeChart.data.datasets[0].data = [];
        emotionTimeChart.update();
    }
    
    if (confidenceChart) {
        confidenceChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
        confidenceChart.update();
    }
    
    if (intensityChart) {
        intensityChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
        intensityChart.update();
    }
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

// ============================================================================
// PAGE LIFECYCLE
// ============================================================================

window.addEventListener('load', () => {
    logToConsole('Application loaded', 'success');
    initializeCharts();
    connectWebSocket();
});

window.addEventListener('beforeunload', () => {
    if (isRecording) {
        stopRecording();
    }
    if (websocket) {
        websocket.close();
    }
});