import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./features.css";
import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { Switch } from "../../components/ui/switch";
import { motion, AnimatePresence } from "framer-motion";
import { Hand, Mic, Smile } from "lucide-react";

// Create axios instance with base URL
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 5000
});

// Create a video feed URL with the correct port
const VIDEO_FEED_URL = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/video_feed`;

export default function Features() {
  const [aslToSpeech, setAslToSpeech] = useState(true);
  const [speechToSign, setSpeechToSign] = useState(false);
  const [expressionAnalysis, setExpressionAnalysis] = useState(false);
  const [recognitionActive, setRecognitionActive] = useState(false);
  const [recognizedText, setRecognizedText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const pollingIntervalRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [avatarVideoUrl, setAvatarVideoUrl] = useState(null);
  const [speechText, setSpeechText] = useState("");
  const speechPollingRef = useRef(null);
  const isListeningRef = useRef(false);
  const accumulatedTextRef = useRef("");

  // Handle mutually exclusive toggles
  const toggleASLToSpeech = () => {
    setAslToSpeech(true);
    setSpeechToSign(false);
  };

  const toggleSpeechToSign = () => {
    setSpeechToSign(true);
    setAslToSpeech(false);
  };

  const startTextPolling = () => {
    stopTextPolling(); // Clear any existing interval first
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const response = await api.get("/api/get_text");
        if (response.data.full_text !== recognizedText) {
          setRecognizedText(response.data.full_text);
          setError(null);
        }
      } catch (error) {
        console.error("Error polling text:", error);
        setError("Error getting recognition text");
      }
    }, 1000); // Poll every second
  };

  const stopTextPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const initializeCamera = async () => {
    try {
      // Skip the getUserMedia check since backend is handling the camera
      return true;
    } catch (error) {
      console.error("Camera permission error:", error);
      return false;
    }
  };

  const startVideoFeed = async () => {
    try {
      // Just verify the backend is accessible
      const response = await fetch(VIDEO_FEED_URL);
      if (!response.ok) {
        throw new Error('Video feed not accessible');
      }
      return true;
    } catch (error) {
      console.error('Error checking video feed:', error);
      setError('Error starting video feed');
      throw error;
    }
  };

  const stopVideoFeed = () => {
    if (videoRef.current) {
      videoRef.current.src = 'about:blank';
    }
  };

  const toggleRecognition = async () => {
    try {
      setIsLoading(true);
      setError(null);

      if (recognitionActive) {
        // First stop the video feed
        stopVideoFeed();
        // Then stop the backend recognition
        await api.post("/api/stop_recognition");
        stopTextPolling();
        setRecognitionActive(false);
      } else {
        try {
          // Check backend health
          const healthCheck = await api.get("/api/video_feed/health");
          
          if (healthCheck.data.status === "healthy" || healthCheck.data.status === "initializing") {
            // Start recognition first
            await api.post("/api/start_recognition");
            
            // Then start video feed and text polling
            await startVideoFeed();
            startTextPolling();
            
            // Set recognition active
            setRecognitionActive(true);
          } else {
            throw new Error("Recognition system not healthy");
          }
        } catch (error) {
          stopVideoFeed();
          stopTextPolling();
          await api.post("/api/stop_recognition").catch(console.error);
          throw error;
        }
      }
    } catch (error) {
      console.error("Error toggling recognition:", error);
      setError("Could not start recognition. Please try again.");
      setRecognitionActive(false);
    } finally {
      setIsLoading(false);
    }
  };

  const stopSpeechPolling = () => {
    if (speechPollingRef.current) {
      clearInterval(speechPollingRef.current);
      speechPollingRef.current = null;
      console.log("Speech polling stopped");
    }
  };

  const startSpeechRecognition = async () => {
    try {
      console.log("Initializing speech recognition...");
      setIsLoading(true);
      setError(null);
      setSpeechText("");
      accumulatedTextRef.current = "";
      setAvatarVideoUrl(null);
      
      // Set both the state and ref
      setIsListening(true);
      isListeningRef.current = true;
      console.log("Set listening state to true");
      
      console.log("Making request to start speech recognition");
      const response = await api.post("/api/start_speech_recognition");
      console.log("Start recognition response:", response.data);
      
      if (response.data.status === "started") {
        console.log("Speech recognition started successfully");
        startSpeechPolling();
      } else {
        console.log("Unexpected response status:", response.data.status);
        setIsListening(false);
        isListeningRef.current = false;
        throw new Error("Failed to start speech recognition");
      }
    } catch (error) {
      console.error("Start recognition error:", error);
      setIsListening(false);
      isListeningRef.current = false;
      const errorMessage = error.response?.data?.detail || 
        "Could not start speech recognition. Please ensure your microphone is connected and permissions are granted.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const startSpeechPolling = () => {
    stopSpeechPolling();
    console.log("Starting speech polling... isListeningRef:", isListeningRef.current);
    
    // Start immediate first poll
    pollSpeechText();
    
    // Then set up interval
    speechPollingRef.current = setInterval(() => {
      if (isListeningRef.current) {
        pollSpeechText();
      } else {
        console.log("Polling interval - not listening anymore");
        stopSpeechPolling();
      }
    }, 1000);
  };

  const pollSpeechText = async () => {
    console.log("Polling - isListeningRef:", isListeningRef.current);
    
    if (!isListeningRef.current) {
        console.log("Not listening anymore (ref check), stopping poll");
        stopSpeechPolling();
        return;
    }
    
    try {
        console.log("Making poll request to /api/get_speech_text");
        const response = await api.get("/api/get_speech_text");
        
        // Log the entire response for debugging
        console.log("Full response from speech text endpoint:", response);
        
        if (response.data && response.data.text) {
            console.log("Got new text:", response.data.text);
            
            // Only add new text if it's not empty
            if (response.data.text.trim()) {
                // Add new text to accumulated text with a space
                accumulatedTextRef.current += (accumulatedTextRef.current ? " " : "") + response.data.text.trim();
                console.log("Updated accumulated text:", accumulatedTextRef.current);
                
                // Update the displayed text
                setSpeechText(accumulatedTextRef.current);
            }
        }
        
        setError(null);
    } catch (error) {
        console.error("Speech polling error:", {
            message: error.message,
            response: error.response?.data,
            status: error.response?.status
        });
        
        if (error.response?.status === 500) {
            setError("Lost connection to speech recognition service. Please try again.");
            await stopSpeechRecognition();
        }
    }
  };

  const stopSpeechRecognition = async () => {
    try {
        setIsLoading(true);
        console.log("Stopping speech recognition...");
        
        // Update both state and ref immediately
        setIsListening(false);
        isListeningRef.current = false;
        console.log("Set listening state to false");
        
        // Stop polling first
        stopSpeechPolling();
        
        // Then stop the backend
        const response = await api.post("/api/stop_speech_recognition");
        console.log("Stop speech recognition response:", response.data);
        
        // Don't clear the text when stopping, keep the accumulated text
        console.log("Final accumulated text:", accumulatedTextRef.current);
    } catch (error) {
        console.error("Error stopping speech recognition:", error);
        setError("Error stopping speech recognition");
    } finally {
        setIsLoading(false);
    }
  };

  useEffect(() => {
    // Cleanup on component unmount
    return () => {
      stopTextPolling();
      stopSpeechPolling();
      stopVideoFeed();
      if (recognitionActive) {
        api.post("/api/stop_recognition").catch(console.error);
      }
      if (isListening) {
        api.post("/api/stop_speech_recognition").catch(console.error);
      }
    };
  }, []);

  const speakText = async () => {
    if (!recognizedText) {
      setError("No text to speak!");
      return;
    }

    try {
      setIsLoading(true);
      const response = await api.post("/api/speak", { text: recognizedText });
      if (response.data.status === "success") {
        const audio = new Audio(`data:audio/mp3;base64,${response.data.audio}`);
        await audio.play();
        setError(null);
      }
    } catch (error) {
      console.error("Error with speak API:", error);
      setError("Error speaking text. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const clearText = async () => {
    try {
      setIsLoading(true);
      await api.post("/api/clear_text");
      setRecognizedText("");
      setError(null);
    } catch (error) {
      console.error("Error clearing text:", error);
      setError("Error clearing text. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const deleteLastCharacter = async () => {
    try {
      setIsLoading(true);
      const response = await api.post("/api/delete_last");
      setRecognizedText(response.data.word_buffer);
      setError(null);
    } catch (error) {
      console.error("Error deleting character:", error);
      setError("Error deleting character. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="features-container">
      <motion.h1
        className="heading"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        AURA
      </motion.h1>

      <Card className="feature-card">
        <CardContent className="feature-card-content">
          <div className="toggle-option">
            <div className="toggle-label">
              <Hand className="icon blue" />
              <span>Enable ASL to Speech</span>
            </div>
            <Switch 
              checked={aslToSpeech} 
              onCheckedChange={toggleASLToSpeech}
              disabled={isLoading} 
            />
          </div>

          <div className="toggle-option">
            <div className="toggle-label">
              <Mic className="icon green" />
              <span>Enable Speech to Sign (3D Avatar)</span>
            </div>
            <Switch 
              checked={speechToSign} 
              onCheckedChange={toggleSpeechToSign}
              disabled={isLoading} 
            />
          </div>

          <div className="toggle-option">
            <div className="toggle-label">
              <Smile className="icon yellow" />
              <span>Enable Facial Expression Analysis</span>
            </div>
            <Switch 
              checked={expressionAnalysis} 
              onCheckedChange={setExpressionAnalysis}
              disabled={isLoading} 
            />
          </div>

          <div className="toggle-option">
            <div className="toggle-label">
              <span>Start/Stop Recognition</span>
            </div>
            <Switch 
              checked={recognitionActive} 
              onCheckedChange={toggleRecognition}
              disabled={isLoading} 
            />
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="feature-content">
        <AnimatePresence mode="wait">
          {aslToSpeech && (
            <motion.div
              key="aslToSpeech"
              className="content-box"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <h2>ASL to Speech</h2>
              <div className="recognized-text">
                {recognizedText || "No text detected yet..."}
              </div>
              <div className="button-group">
                <Button 
                  onClick={speakText} 
                  disabled={!recognizedText || isLoading}
                >
                  Speak Text
                </Button>
                <Button 
                  onClick={clearText} 
                  disabled={!recognizedText || isLoading}
                >
                  Clear Text
                </Button>
                <Button 
                  onClick={deleteLastCharacter} 
                  disabled={!recognizedText || isLoading}
                >
                  Delete Last
                </Button>
              </div>
            </motion.div>
          )}

          {speechToSign && (
            <motion.div
              key="speechToSign"
              className="content-box"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <h2>Speech to Sign (3D Avatar)</h2>
              <div className="speech-text">
                {speechText || "No speech detected yet..."}
              </div>
              <div className="avatar-container">
                {avatarVideoUrl ? (
                  <video
                    className="avatar-video"
                    src={avatarVideoUrl}
                    autoPlay
                    loop
                    muted
                    playsInline
                  />
                ) : (
                  <div className="avatar-placeholder">
                    {isListening ? "Listening for speech..." : "Click Start to begin"}
                  </div>
                )}
              </div>
              <div className="button-group">
                <Button
                  onClick={isListening ? stopSpeechRecognition : startSpeechRecognition}
                  disabled={isLoading}
                >
                  {isListening ? "Stop Listening" : "Start Listening"}
                </Button>
              </div>
            </motion.div>
          )}

          {recognitionActive && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="video-feed-container"
              style={{
                width: "640px",
                height: "680px",
                position: "relative",
                overflow: "hidden",
                backgroundColor: "#000",
                borderRadius: "8px",
                marginBottom: "1rem"
              }}
            >
              <iframe
                ref={videoRef}
                className="video-feed"
                src={VIDEO_FEED_URL}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  border: "none",
                  backgroundColor: "transparent"
                }}
                title="ASL Recognition Feed"
                scrolling="no"
              />
              {isLoading && (
                <div className="loading-overlay">
                  <span>Loading camera feed...</span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {recognitionActive && (
          <div className="status-message">
            Recognition is active - detecting signs...
          </div>
        )}
      </div>
    </div>
  );
}

