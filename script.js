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
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fuzzyMatchToggle = document.getElementById('fuzzyMatchToggle');

// State
let isRecording = false;
let recordingStartTime = null;
let timerInterval = null;
let websocket = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
let fuzzyMatchEnabled = true;

// WAV Recording variables
let audioContext = null;
let source = null;
let processor = null;
let stream = null;
let pcmData = [];

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

// ============================================================================
// FUZZY MATCH TOGGLE
// ============================================================================

fuzzyMatchToggle.addEventListener('change', (e) => {
    fuzzyMatchEnabled = e.target.checked;
    const status = fuzzyMatchEnabled ? 'enabled' : 'disabled';
    logToConsole(`🔍 Fuzzy matching ${status}`, 'info');
    
    // Send toggle state to server
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'fuzzy_match_toggle',
            enabled: fuzzyMatchEnabled
        }));
    }
});

// ============================================================================
// SERVER RESPONSE HANDLER
// ============================================================================

function handleServerResponse(data) {
    console.log('handleServerResponse called with:', data);
    
    // Handle errors
    if (data.error) {
        logToConsole(`✗ Server error: ${data.error}`, 'error');
        statusDisplay.textContent = 'Error - Click to retry';
        return;
    }
    
    // Handle no speech detected
    if (data.status === 'no_speech') {
        logToConsole('No speech detected in audio', 'warning');
        statusDisplay.textContent = 'No speech detected';
        setTimeout(() => {
            statusDisplay.textContent = 'Click microphone to start';
        }, 2000);
        return;
    }
    
    // Update transcription
    if (data.konkani) {
        konkaniText.textContent = data.konkani;
        logToConsole(`Konkani: ${data.konkani}`, 'success');
        
        // Show fuzzy match info if available
        if (data.fuzzy_match) {
            const matchInfo = `(Matched with ${data.fuzzy_match.similarity_score.toFixed(1)}% similarity)`;
            logToConsole(`🔍 Fuzzy match: ${matchInfo}`, 'info');
            if (data.konkani_transcribed && data.konkani_transcribed !== data.konkani) {
                logToConsole(`Original: ${data.konkani_transcribed}`, 'info');
            }
        }
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
    
    // Reset status after successful processing
    statusDisplay.textContent = 'Click microphone to start';
    logToConsole('✓ Results displayed successfully', 'success');
}

// ============================================================================
// EMOTION DISPLAY
// ============================================================================

function updateEmotion(emotionScores) {
    console.log('Updating emotion display with:', emotionScores);
    
    // emotionScores is a dict like: {neutral: 25.0, happy: 50.0, sad: 15.0, angry: 10.0}
    
    // Find dominant emotion
    let dominantEmotion = 'neutral';
    let maxScore = 0;
    
    for (const [emotion, score] of Object.entries(emotionScores)) {
        if (score > maxScore) {
            maxScore = score;
            dominantEmotion = emotion;
        }
    }
    
    // Update badge
    emotionBadge.textContent = dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1);
    emotionBadge.className = 'emotion-badge emotion-' + dominantEmotion.toLowerCase();
    emotionConfidence.textContent = `${maxScore.toFixed(1)}% confidence`;
    
    // Update emotion bars
    updateEmotionBars(emotionScores);
    
    logToConsole(`Emotion: ${dominantEmotion} (${maxScore.toFixed(1)}%)`, 'info');
}

function updateEmotionBars(scores) {
    // Update each emotion bar
    const emotions = ['happy', 'sad', 'angry', 'neutral'];
    
    emotions.forEach(emotion => {
        const score = scores[emotion] || 0;
        const barElement = document.getElementById(`${emotion}Bar`);
        const valueElement = document.getElementById(`${emotion}Value`);
        
        if (barElement && valueElement) {
            // Animate the bar
            barElement.style.width = `${score}%`;
            valueElement.textContent = `${score.toFixed(1)}%`;
        }
    });
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
        // Request microphone access only once
        if (!stream || !stream.active) {
            logToConsole('🎤 Requesting microphone access...', 'info');
            
            stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: false,
                    autoGainControl: true
                } 
            });
            
            // Log the actual audio track settings
            const audioTrack = stream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            logToConsole(`✓ Microphone: ${audioTrack.label}`, 'success');
            logToConsole(`  Sample Rate: ${settings.sampleRate}Hz, Channels: ${settings.channelCount}`, 'info');
        }
        
        // Create AudioContext with 16kHz sample rate (reuse if exists)
        if (!audioContext || audioContext.state === 'closed') {
            audioContext = new AudioContext({ sampleRate: 16000 });
            logToConsole(`✓ Recording format: WAV (16kHz, 16-bit)`, 'success');
        }

        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        pcmData = [];

        processor.onaudioprocess = e => {
            pcmData.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };

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
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (source) {
        source.disconnect();
        source = null;
    }
    // Keep stream and audioContext alive for next recording

    logToConsole('⏹ Recording stopped - Processing audio...', 'warning');
    isRecording = false;
    micButton.classList.remove('recording');
    statusDisplay.textContent = 'Processing...';
    statusDisplay.classList.remove('recording');
    
    stopTimer();
    stopVisualizer();

    // Convert PCM to WAV and send to server
    if (pcmData.length > 0) {
        const wavBlob = pcmToWav(pcmData, 16000);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileName = `recording-${timestamp}.wav`;
        
        logToConsole(`✓ Created WAV file: ${fileName} (${wavBlob.size} bytes)`, 'success');
        
        // Send to server if connected
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            logToConsole(`📤 Sending WAV file to server...`, 'info');
            websocket.send(wavBlob);
        } else {
            showError('Server not connected. Please wait for connection.');
            logToConsole(`✗ Cannot send file - server disconnected`, 'error');
            statusDisplay.textContent = 'Click microphone to start';
        }
        
        // Clear PCM data
        pcmData = [];
    } else {
        logToConsole('⚠ No audio recorded', 'warning');
        statusDisplay.textContent = 'Click microphone to start';
    }
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
// WAV ENCODER
// ============================================================================

function pcmToWav(pcmChunks, sampleRate) {
    const samples = mergePCM(pcmChunks);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function mergePCM(chunks) {
    let length = 0;
    chunks.forEach(c => length += c.length);
    const result = new Float32Array(length);
    let offset = 0;
    chunks.forEach(c => {
        result.set(c, offset);
        offset += c.length;
    });
    return result;
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

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
// DRAG AND DROP
// ============================================================================

function setupDragAndDrop() {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    
    // Handle click to browse
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Check if it's a WAV file
    if (!file.type.includes('audio/wav') && !file.name.toLowerCase().endsWith('.wav')) {
        showError('Please upload a WAV audio file');
        logToConsole('✗ Invalid file type. Only WAV files are supported.', 'error');
        return;
    }

    logToConsole(`📁 File selected: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`, 'info');
    statusDisplay.textContent = 'Processing file...';

    // Read the file and send to server
    const reader = new FileReader();
    
    reader.onload = async (e) => {
        const arrayBuffer = e.target.result;
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            logToConsole('📤 Sending file to server...', 'info');
            websocket.send(blob);
            statusDisplay.textContent = 'Analyzing audio...';
        } else {
            showError('Server not connected. Please wait for connection.');
            logToConsole('✗ Cannot send file - server disconnected', 'error');
            statusDisplay.textContent = 'Click microphone to start';
        }
    };
    
    reader.onerror = () => {
        showError('Failed to read file');
        logToConsole('✗ File read error', 'error');
        statusDisplay.textContent = 'Click microphone to start';
    };
    
    reader.readAsArrayBuffer(file);
}

// ============================================================================
// PAGE LIFECYCLE
// ============================================================================

window.addEventListener('load', () => {
    logToConsole('Application loaded', 'success');
    initializeCharts();
    connectWebSocket();
    setupDragAndDrop();
});

window.addEventListener('beforeunload', () => {
    if (isRecording) {
        stopRecording();
    }
    // Clean up microphone and audio resources
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    if (websocket) {
        websocket.close();
    }
});