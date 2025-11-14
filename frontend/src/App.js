import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Loader, AlertCircle, CheckCircle, Sparkles, ArrowRight } from 'lucide-react';

function App() {
  const [page, setPage] = useState('home'); // 'home', 'scanner', or 'scanning'
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [scanProgress, setScanProgress] = useState(0);
  
  // Medical Q&A Chat states
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);

  // Add CSS for animations
  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    @keyframes gradient-shift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-20px); }
    }
    
    @keyframes glow {
      0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
      50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.6); }
    }
    
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .gradient-bg {
      background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0a0a0a 100%);
      font-family: 'Inter', sans-serif;
      position: relative;
    }
    
    .gradient-bg::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at 20% 50%, rgba(100, 200, 255, 0.1) 0%, transparent 50%),
                  radial-gradient(circle at 80% 80%, rgba(150, 100, 255, 0.1) 0%, transparent 50%);
      pointer-events: none;
    }
    
    .float-animation {
      animation: float 6s ease-in-out infinite;
    }
    
    .glow-button {
      animation: glow 2s ease-in-out infinite;
    }
    
    .slide-in {
      animation: slideIn 0.5s ease-out;
    }
    
    .button-hover {
      transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .button-hover:hover {
      transform: translateY(-2px);
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .button-hover:active {
      transform: translateY(0px);
    }
    
    .title-font {
      font-family: 'Poppins', sans-serif;
      font-weight: 800;
      letter-spacing: -0.5px;
    }
    
    .heading-font {
      font-family: 'Poppins', sans-serif;
      font-weight: 700;
    }
    
    .text-gradient {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    @keyframes pulse-glow {
      0%, 100% { opacity: 0.5; transform: scale(1); }
      50% { opacity: 1; transform: scale(1.05); }
    }
    
    @keyframes rotate-slow {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    
    @keyframes bounce-gentle {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fadeIn {
      animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes scan-line {
      0% { transform: translateY(-100%); }
      100% { transform: translateY(100%); }
    }
    
    .pulse-glow {
      animation: pulse-glow 3s ease-in-out infinite;
    }
    
    .rotate-slow {
      animation: rotate-slow 20s linear infinite;
    }
    
    .bounce-gentle {
      animation: bounce-gentle 2s ease-in-out infinite;
    }
    
    .scan-line {
      animation: scan-line 2s ease-in-out infinite;
    }
    
    .feature-card {
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    
    .feature-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 10px 30px rgba(100, 200, 255, 0.2);
    }
    
    .feature-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.1), transparent);
      transition: left 0.5s;
    }
    
    .feature-card:hover::before {
      left: 100%;
    }
    
    @keyframes result-appear {
      0% {
        opacity: 0;
        transform: translateY(20px) scale(0.95);
      }
      100% {
        opacity: 1;
        transform: translateY(0) scale(1);
      }
    }
    
    @keyframes shimmer {
      0% {
        background-position: -1000px 0;
      }
      100% {
        background-position: 1000px 0;
      }
    }
    
    @keyframes border-glow {
      0%, 100% {
        box-shadow: 0 0 10px rgba(100, 200, 255, 0.3), inset 0 0 10px rgba(100, 200, 255, 0.1);
      }
      50% {
        box-shadow: 0 0 20px rgba(100, 200, 255, 0.6), inset 0 0 20px rgba(100, 200, 255, 0.2);
      }
    }
    
    .result-card {
      animation: result-appear 0.6s ease-out, border-glow 3s ease-in-out infinite;
      background: linear-gradient(135deg, rgba(50, 50, 50, 0.8) 0%, rgba(40, 40, 40, 0.8) 100%);
      position: relative;
      overflow: hidden;
    }
    
    .result-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: -1000px;
      width: 1000px;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.2), transparent);
      animation: shimmer 3s infinite;
    }
    
    .result-content {
      position: relative;
      z-index: 1;
    }
    
    .result-text {
      background: linear-gradient(135deg, #b0b0b0 0%, #e8e8e8 50%, #b0b0b0 100%);
      background-size: 200% 200%;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes scan-pulse {
      0% { transform: scale(0.8); opacity: 0; }
      50% { opacity: 1; }
      100% { transform: scale(1.2); opacity: 0; }
    }
    
    @keyframes scan-beam {
      0% { top: -100%; }
      100% { top: 100%; }
    }
    
    @keyframes progress-fill {
      0% { width: 0%; }
      100% { width: 100%; }
    }
    
    .scan-circle {
      animation: scan-pulse 1s ease-out infinite;
    }
    
    .scan-beam {
      animation: scan-beam 2s ease-in-out infinite;
    }
    
    .progress-bar {
      animation: progress-fill 3s ease-out forwards;
    }
    
    @keyframes result-glow {
      0%, 100% { box-shadow: 0 0 20px rgba(100, 200, 255, 0.3), inset 0 0 20px rgba(100, 200, 255, 0.1); }
      50% { box-shadow: 0 0 40px rgba(100, 200, 255, 0.6), inset 0 0 30px rgba(100, 200, 255, 0.2); }
    }
    
    @keyframes result-slide {
      0% { opacity: 0; transform: translateY(30px) scale(0.95); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    @keyframes text-shimmer {
      0% { background-position: -1000px 0; }
      100% { background-position: 1000px 0; }
    }
    
    .result-card-enhanced {
      animation: result-glow 3s ease-in-out infinite, result-slide 0.8s ease-out;
      background: linear-gradient(135deg, rgba(30, 30, 30, 0.95) 0%, rgba(20, 20, 20, 0.95) 100%);
      position: relative;
      overflow: hidden;
    }
    
    .result-card-enhanced::before {
      content: '';
      position: absolute;
      top: 0;
      left: -1000px;
      width: 1000px;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.2), transparent);
      animation: text-shimmer 3s infinite;
    }
    
    .result-header {
      background: linear-gradient(135deg, rgba(100, 200, 255, 0.1) 0%, rgba(150, 100, 255, 0.1) 100%);
      border-bottom: 2px solid rgba(100, 200, 255, 0.3);
      padding: 1rem;
      border-radius: 0.75rem 0.75rem 0 0;
      margin: -1.5rem -1.5rem 1rem -1.5rem;
    }
    
    .result-content-enhanced {
      position: relative;
      z-index: 1;
    }
  `;

  const handleStartScanning = () => {
    setPage('scanning');
    setScanProgress(0);
    
    // Simulate progress
    const interval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + Math.random() * 30;
      });
    }, 300);

    // After 3 seconds, go to scanner page
    setTimeout(() => {
      clearInterval(interval);
      setPage('scanner');
      setScanProgress(0);
    }, 3000);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    setLoading(true);
    setError(null);
    setResult(null);
    setScanProgress(10);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      console.log('API URL:', apiUrl);
      
      setScanProgress(50);
      
      const response = await axios.post(`${apiUrl}/api/analyze`, formData, {
        timeout: 60000, // 60 second timeout - full ML model takes longer
        // Don't set Content-Type header - axios will set it automatically with boundary
      });
      
      setScanProgress(100);
      
      if (response.data && response.data.result) {
        setResult(response.data.result);
        setError(null);
      } else {
        setError('No analysis result received from server');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setScanProgress(0);
      
      if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.');
      } else if (err.response?.status === 404) {
        setError('API endpoint not found.');
      } else if (err.response?.status === 500) {
        setError('Server error. Please try again later.');
      } else if (!err.response) {
        setError('Cannot connect to server. Check your internet connection.');
      } else {
        setError(err.response?.data?.error || 'Analysis failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  // Medical Q&A Handler
  const handleAskMedical = async () => {
    if (!chatInput.trim()) return;
    
    setChatError(null);
    
    // Add user message to chat
    const userMessage = {
      type: 'user',
      text: chatInput,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      
      const response = await axios.post(
        `${apiUrl}/api/ask-medical`,
        {
          question: userMessage.text,
          disease: null // Could be extracted from result later
        },
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );

      if (response.data.success) {
        const assistantMessage = {
          type: 'assistant',
          text: response.data.answer,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(response.data.error || 'Failed to get response');
      }
    } catch (err) {
      const errorMsg = err.response?.data?.error || err.message || 'Error getting medical information';
      setChatError(errorMsg);
      
      // Add error message to chat
      const errorMessage = {
        type: 'error',
        text: `Error: ${errorMsg}`,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <>
      <style>{styles}</style>
      <div className="gradient-bg min-h-screen py-8 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute top-10 left-10 w-72 h-72 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 float-animation"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 float-animation" style={{animationDelay: '2s'}}></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 float-animation" style={{animationDelay: '4s'}}></div>
        
        <div className="max-w-4xl mx-auto relative z-10">
          {/* Header */}
          <div className="mb-8 slide-in relative z-10 flex items-center justify-center gap-8">
            {/* Logo - Left Side */}
            <div className="flex-shrink-0 slide-in" style={{animationDelay: '0.2s'}}>
              <div className="w-32 h-32 rounded-full flex items-center justify-center" style={{
                background: 'linear-gradient(135deg, rgba(100, 200, 255, 0.1) 0%, rgba(150, 100, 255, 0.1) 100%)',
                border: '2px solid rgba(100, 200, 255, 0.3)',
                boxShadow: '0 0 40px rgba(100, 200, 255, 0.3), inset 0 0 20px rgba(100, 200, 255, 0.1)',
                animation: 'float 6s ease-in-out infinite'
              }}>
                <img src="/logo.svg" alt="MediScanner AI Logo" className="w-24 h-24" />
              </div>
            </div>

            {/* Title - Right Side */}
            <div className="text-left flex-1">
              <h1 className="text-5xl sm:text-6xl font-bold drop-shadow-lg title-font mb-2" style={{
                background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textShadow: '0 0 30px rgba(100, 200, 255, 0.2)',
                letterSpacing: '-1px',
                lineHeight: '1.2',
                paddingBottom: '0.5rem'
              }}>
                MediScanner AI
              </h1>
              <p className="text-lg drop-shadow-md opacity-90" style={{fontFamily: "'Inter', sans-serif", fontWeight: '500', color: '#b0b0b0'}}>
                AI-powered analysis of medical images using advanced imaging expertise
              </p>
            </div>
          </div>

          {/* Home Page */}
          {page === 'home' && (
            <div className="space-y-8">
              {/* Main Content Grid */}
              <div className="flex flex-col lg:flex-row gap-20 items-start">
                {/* Right Side - Main Description */}
                <div className="w-full flex-1">
                  {/* Description Card */}
                  <div className="backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden slide-in border p-8" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(192, 192, 192, 0.2)'}}>
                    <div>
                      <h2 className="text-2xl font-bold mb-4 heading-font" style={{color: '#e8e8e8'}}>About This Application</h2>
                      
                      <div className="space-y-3 text-base leading-relaxed" style={{fontFamily: "'Inter', sans-serif"}}>
                        <p style={{color: '#b0b0b0'}}>
                          Welcome to <span className="font-semibold" style={{background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>MediScanner AI</span> - an advanced AI-powered platform designed to assist healthcare professionals in analyzing medical images with precision and speed.
                        </p>
                        
                        <p style={{color: '#b0b0b0'}}>
                          Our system leverages a <span style={{color: '#64c8ff', fontWeight: 'bold'}}>specialized CT/MRI Binary Classifier</span> to accurately distinguish between CT and MRI imaging modalities. Built with advanced deep learning architecture and trained on diverse medical imaging datasets, our platform provides <span style={{color: '#64c8ff', fontWeight: 'bold'}}>99.66% accuracy</span> in imaging type classification.
                        </p>
                        
                        <div className="p-6 rounded-xl border my-6" style={{backgroundColor: 'rgba(50, 50, 50, 0.8)', borderColor: 'rgba(192, 192, 192, 0.3)'}}>
                          <h3 className="font-semibold mb-3 heading-font" style={{color: '#64c8ff', fontSize: '1.1rem'}}>🚀 Advanced Features:</h3>
                          <ul className="space-y-2" style={{fontFamily: "'Inter', sans-serif"}}>
                            <li className="flex items-center gap-2" style={{color: '#b0b0b0'}}>
                              <span style={{color: '#64c8ff'}}>✓</span> CT/MRI Binary Classification with 99.66% accuracy
                            </li>
                            <li className="flex items-center gap-2" style={{color: '#b0b0b0'}}>
                              <span style={{color: '#64c8ff'}}>✓</span> 1.4M+ parameters deep learning model
                            </li>
                            <li className="flex items-center gap-2" style={{color: '#b0b0b0'}}>
                              <span style={{color: '#64c8ff'}}>✓</span> Advanced CNN architecture with batch normalization
                            </li>
                            <li className="flex items-center gap-2" style={{color: '#b0b0b0'}}>
                              <span style={{color: '#64c8ff'}}>✓</span> Real-time imaging modality detection
                            </li>
                            <li className="flex items-center gap-2" style={{color: '#b0b0b0'}}>
                              <span style={{color: '#64c8ff'}}>✓</span> Instant analysis with confidence scoring
                            </li>
                          </ul>
                        </div>
                        
                        <p style={{color: '#b0b0b0'}}>
                          Simply upload your medical image and let our AI analyze it. You'll receive detailed findings highlighting key observations and potential diagnoses to support your clinical decision-making. Our models are trained on real medical datasets and provide clinical-grade accuracy.
                        </p>
                      </div>
                    </div>
                  </div>

                </div>
              </div>

              {/* Features Grid with Animations */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Feature 1: Brain Analysis */}
                <div className="feature-card backdrop-blur-md rounded-2xl border p-6 slide-in" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)'}}>
                  <div className="flex flex-col items-center text-center">
                    <div className="mb-4 pulse-glow" style={{fontSize: '3rem'}}>🧠</div>
                    <h3 className="text-xl font-bold mb-2 heading-font" style={{color: '#64c8ff'}}>Brain Imaging</h3>
                    <p style={{color: '#b0b0b0'}}>Advanced neural analysis and detection</p>
                  </div>
                </div>

                {/* Feature 2: Scan Technology */}
                <div className="feature-card backdrop-blur-md rounded-2xl border p-6 slide-in" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)', animationDelay: '0.1s'}}>
                  <div className="flex flex-col items-center text-center">
                    <div className="mb-4 bounce-gentle" style={{fontSize: '3rem'}}>🔬</div>
                    <h3 className="text-xl font-bold mb-2 heading-font" style={{color: '#9664ff'}}>Scan Tech</h3>
                    <p style={{color: '#b0b0b0'}}>Multi-modal imaging support</p>
                  </div>
                </div>

                {/* Feature 3: AI Analysis */}
                <div className="feature-card backdrop-blur-md rounded-2xl border p-6 slide-in" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)', animationDelay: '0.2s'}}>
                  <div className="flex flex-col items-center text-center">
                    <div className="mb-4 rotate-slow" style={{fontSize: '3rem'}}>⚡</div>
                    <h3 className="text-xl font-bold mb-2 heading-font" style={{color: '#64c8ff'}}>AI Powered</h3>
                    <p style={{color: '#b0b0b0'}}>Real-time intelligent analysis</p>
                  </div>
                </div>
              </div>

              {/* Data Science Impact Section */}
              <div className="mt-12 slide-in" style={{animationDelay: '0.3s'}}>
                <h2 className="text-3xl font-bold mb-8 heading-font text-center" style={{color: '#e8e8e8'}}>🔬 Data Science & ML Impact</h2>
                
                {/* ML Models Section - Custom + Pretrained */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  {/* CT/MRI Binary Classifier - Custom Trained */}
                  <div className="backdrop-blur-md rounded-2xl border p-6 feature-card" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)'}}>
                    <div className="flex items-center gap-3 mb-4">
                      <div style={{fontSize: '2rem'}}>🎯</div>
                      <div>
                        <h3 className="text-xl font-bold heading-font" style={{color: '#64c8ff'}}>CT/MRI Classifier</h3>
                        <p style={{color: '#808080', fontSize: '0.75rem'}}>Custom Trained Model</p>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Type: <span style={{color: '#64c8ff', fontWeight: 'bold'}}>CNN Binary</span></p>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Parameters: <span style={{color: '#64c8ff', fontWeight: 'bold'}}>1.4M</span></p>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Input: <span style={{color: '#64c8ff', fontWeight: 'bold'}}>224×224×3</span></p>
                      </div>
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', marginBottom: '0.5rem'}}>Accuracy</p>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div className="bg-gradient-to-r from-cyan-400 to-blue-500 h-2 rounded-full" style={{width: '99.66%'}}></div>
                        </div>
                        <p style={{color: '#64c8ff', fontSize: '0.75rem', marginTop: '0.25rem'}}>99.66%</p>
                      </div>
                      <div className="pt-2 border-t" style={{borderColor: 'rgba(100, 200, 255, 0.2)'}}>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '0.5rem'}}>Purpose:</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>🔵 CT vs 🟣 MRI Detection</p>
                      </div>
                    </div>
                  </div>

                  {/* Pretrained Models Info */}
                  <div className="backdrop-blur-md rounded-2xl border p-6 feature-card" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(150, 100, 255, 0.3)'}}>
                    <div className="flex items-center gap-3 mb-4">
                      <div style={{fontSize: '2rem'}}>🧠</div>
                      <div>
                        <h3 className="text-xl font-bold heading-font" style={{color: '#9664ff'}}>Pretrained Models</h3>
                        <p style={{color: '#808080', fontSize: '0.75rem'}}>ImageNet Base + Medical Fine-tuning</p>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '0.5rem'}}>🫁 DenseNet121</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• ImageNet weights base</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Medical fine-tuning ready</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• 7.9M parameters</p>
                      </div>
                      <div className="pt-2 border-t" style={{borderColor: 'rgba(150, 100, 255, 0.2)'}}>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '0.5rem'}}>🏥 MobileNetV2</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• ImageNet weights base</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Lightweight & fast</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• 3.5M parameters</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Performance & Architecture */}
                <div className="backdrop-blur-md rounded-2xl border p-8 slide-in" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)', animationDelay: '0.2s'}}>
                  <h3 className="text-2xl font-bold mb-6 heading-font" style={{color: '#64c8ff'}}>🚀 Model Performance & Architecture</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    {/* Classification Accuracy */}
                    <div>
                      <p style={{color: '#b0b0b0', fontSize: '0.875rem', marginBottom: '0.5rem'}}>Classification Accuracy</p>
                      <div className="text-3xl font-bold heading-font" style={{color: '#64c8ff', marginBottom: '0.5rem'}}>99.66%</div>
                      <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>CT vs MRI Detection</p>
                    </div>

                    {/* Processing Speed */}
                    <div>
                      <p style={{color: '#b0b0b0', fontSize: '0.875rem', marginBottom: '0.5rem'}}>Inference Speed</p>
                      <div className="text-3xl font-bold heading-font" style={{color: '#9664ff', marginBottom: '0.5rem'}}>~100ms</div>
                      <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>Per image analysis</p>
                    </div>

                    {/* Model Size */}
                    <div>
                      <p style={{color: '#b0b0b0', fontSize: '0.875rem', marginBottom: '0.5rem'}}>Model Parameters</p>
                      <div className="text-3xl font-bold heading-font" style={{color: '#64c8ff', marginBottom: '0.5rem'}}>1.4M</div>
                      <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>Optimized CNN</p>
                    </div>
                  </div>

                  {/* Architecture Information */}
                  <div className="mb-6 p-4 rounded-lg" style={{backgroundColor: 'rgba(100, 200, 255, 0.1)', borderLeft: '4px solid #64c8ff'}}>
                    <h4 className="font-bold mb-3" style={{color: '#64c8ff'}}>🏗️ Model Architecture:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold'}}>Input Layer</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• 224×224×3 RGB images</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Normalized pixel values</p>
                      </div>
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold'}}>Output Layer</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Binary classification (CT/MRI)</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Sigmoid activation</p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t" style={{borderColor: 'rgba(100, 200, 255, 0.2)'}}>
                    <h4 className="font-bold mb-3" style={{color: '#64c8ff'}}>🧠 Advanced Features:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="flex items-start gap-2">
                        <span style={{color: '#64c8ff'}}>✓</span>
                        <span style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Data augmentation (flips, rotations, zoom)</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span style={{color: '#64c8ff'}}>✓</span>
                        <span style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Batch normalization for stability</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span style={{color: '#64c8ff'}}>✓</span>
                        <span style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Dropout regularization</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span style={{color: '#64c8ff'}}>✓</span>
                        <span style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Global average pooling</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span style={{color: '#64c8ff'}}>✓</span>
                        <span style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Confidence scoring</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span style={{color: '#64c8ff'}}>✓</span>
                        <span style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Real-time predictions</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* How Models Impact Output */}
                <div className="backdrop-blur-md rounded-2xl border p-8 slide-in" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)', animationDelay: '0.3s'}}>
                  <h3 className="text-2xl font-bold mb-6 heading-font" style={{color: '#64c8ff'}}>📊 How Models Impact Output</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Custom CT/MRI Model Impact */}
                    <div className="p-4 rounded-lg" style={{backgroundColor: 'rgba(100, 200, 255, 0.1)', borderLeft: '4px solid #64c8ff'}}>
                      <h4 className="font-bold mb-3" style={{color: '#64c8ff'}}>🎯 CT/MRI Classifier</h4>
                      <div className="space-y-2">
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>
                          <span style={{color: '#64c8ff', fontWeight: 'bold'}}>Primary Output:</span>
                        </p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Imaging modality classification</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Confidence scores (0-100%)</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• CT vs MRI detection</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem', marginTop: '0.5rem'}}>
                          <span style={{color: '#64c8ff', fontWeight: 'bold'}}>Impact:</span> Determines scan type for specialized analysis
                        </p>
                      </div>
                    </div>

                    {/* DenseNet121 Impact */}
                    <div className="p-4 rounded-lg" style={{backgroundColor: 'rgba(150, 100, 255, 0.1)', borderLeft: '4px solid #9664ff'}}>
                      <h4 className="font-bold mb-3" style={{color: '#9664ff'}}>🫁 DenseNet121</h4>
                      <div className="space-y-2">
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>
                          <span style={{color: '#9664ff', fontWeight: 'bold'}}>Secondary Output:</span>
                        </p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Feature extraction (7.9M params)</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Dense connections</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Medical pattern recognition</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem', marginTop: '0.5rem'}}>
                          <span style={{color: '#9664ff', fontWeight: 'bold'}}>Impact:</span> Enhances feature detection & accuracy
                        </p>
                      </div>
                    </div>

                    {/* MobileNetV2 Impact */}
                    <div className="p-4 rounded-lg" style={{backgroundColor: 'rgba(100, 255, 150, 0.1)', borderLeft: '4px solid #64ff96'}}>
                      <h4 className="font-bold mb-3" style={{color: '#64ff96'}}>🏥 MobileNetV2</h4>
                      <div className="space-y-2">
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>
                          <span style={{color: '#64ff96', fontWeight: 'bold'}}>Tertiary Output:</span>
                        </p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Fast inference (3.5M params)</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Lightweight processing</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Real-time analysis</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem', marginTop: '0.5rem'}}>
                          <span style={{color: '#64ff96', fontWeight: 'bold'}}>Impact:</span> Ensures speed & efficiency
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Combined Effect */}
                  <div className="mt-6 p-4 rounded-lg" style={{backgroundColor: 'rgba(100, 200, 255, 0.15)', border: '2px solid rgba(100, 200, 255, 0.3)'}}>
                    <h4 className="font-bold mb-3" style={{color: '#64c8ff'}}>🚀 Combined Effect:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '0.5rem'}}>Accuracy Improvement</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Multi-model ensemble approach</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Cross-validation of results</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Reduced false positives</p>
                      </div>
                      <div>
                        <p style={{color: '#b0b0b0', fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '0.5rem'}}>Performance Metrics</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• ~100ms total inference time</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• 99.66% classification accuracy</p>
                        <p style={{color: '#b0b0b0', fontSize: '0.75rem'}}>• Optimal accuracy-speed tradeoff</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Get Started Button - At Bottom */}
              <div className="mt-12 text-center slide-in" style={{animationDelay: '0.4s'}}>
                <button
                  onClick={handleStartScanning}
                  className="button-hover text-white font-bold py-4 px-12 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-3 text-lg mx-auto"
                  style={{background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)', boxShadow: '0 0 30px rgba(100, 200, 255, 0.4)'}}
                >
                  <span>Get Started</span>
                  <ArrowRight className="w-6 h-6" />
                </button>
              </div>
            </div>
          )}

          {/* Scanning Animation Page */}
          {page === 'scanning' && (
            <div className="flex items-center justify-center min-h-screen">
              <div className="text-center">
                {/* Scanning Circle Animation */}
                <div className="relative w-40 h-40 mx-auto mb-8">
                  {/* Outer circle */}
                  <div className="absolute inset-0 rounded-full border-4" style={{borderColor: 'rgba(100, 200, 255, 0.2)'}}></div>
                  
                  {/* Middle circle */}
                  <div className="absolute inset-4 rounded-full border-4" style={{borderColor: 'rgba(100, 200, 255, 0.4)'}}></div>
                  
                  {/* Inner pulsing circle */}
                  <div className="absolute inset-8 rounded-full border-4 scan-circle" style={{borderColor: '#64c8ff'}}></div>
                  
                  {/* Scanning beam */}
                  <div className="absolute inset-0 scan-beam" style={{
                    background: 'linear-gradient(180deg, transparent, rgba(100, 200, 255, 0.3), transparent)',
                    borderRadius: '50%'
                  }}></div>
                  
                  {/* Center dot */}
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full" style={{background: '#64c8ff'}}></div>
                </div>

                {/* Text */}
                <h2 className="text-3xl font-bold heading-font mb-4" style={{color: '#e8e8e8'}}>Initializing Scanner</h2>
                <p style={{color: '#b0b0b0', marginBottom: '2rem'}}>Preparing medical imaging analysis...</p>

                {/* Progress Bar */}
                <div className="w-64 h-2 bg-gray-700 rounded-full mx-auto overflow-hidden">
                  <div 
                    className="h-full progress-bar" 
                    style={{
                      background: 'linear-gradient(90deg, #64c8ff 0%, #9664ff 100%)',
                      width: `${Math.min(scanProgress, 100)}%`
                    }}
                  ></div>
                </div>
                
                {/* Progress Text */}
                <p style={{color: '#64c8ff', marginTop: '1rem', fontWeight: 'bold'}}>
                  {Math.min(Math.round(scanProgress), 100)}%
                </p>

                {/* Loading dots */}
                <div className="mt-8 flex justify-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{background: '#64c8ff', animation: 'pulse 1.4s infinite'}}></div>
                  <div className="w-2 h-2 rounded-full" style={{background: '#64c8ff', animation: 'pulse 1.4s infinite 0.2s'}}></div>
                  <div className="w-2 h-2 rounded-full" style={{background: '#64c8ff', animation: 'pulse 1.4s infinite 0.4s'}}></div>
                </div>
              </div>
            </div>
          )}

          {/* Scanner Page */}
          {page === 'scanner' && (
            <>
              {/* Back Button */}
              <button
                onClick={() => setPage('home')}
                className="mb-6 hover:opacity-80 transition-opacity flex items-center gap-2 drop-shadow-lg"
                style={{color: '#b0b0b0'}}
              >
                <span>← Back</span>
              </button>

              {/* Main Container - Structured Flow */}
              <div className="space-y-6">
                
                {/* STEP 1: INPUT SECTION */}
                <div className="backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden slide-in border p-8" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)'}}>
                  <div className="flex items-center gap-3 mb-6">
                    <div style={{fontSize: '1.5rem', background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>📤</div>
                    <h2 className="text-2xl font-bold heading-font" style={{color: '#e8e8e8'}}>Step 1: Upload Image</h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Upload Area */}
                    <div className="md:col-span-1">
                      <label className="block text-sm font-semibold mb-4" style={{color: '#b0b0b0'}}>
                        Select Medical Image
                      </label>
                      <div className="relative">
                        <input
                          type="file"
                          accept=".png,.jpg,.jpeg,.dicom"
                          onChange={handleFileChange}
                          disabled={loading}
                          className="hidden"
                          id="file-input"
                        />
                        <label
                          htmlFor="file-input"
                          className={`flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-lg cursor-pointer transition-colors relative ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                          style={{
                            borderColor: preview ? '#64c8ff' : 'rgba(192, 192, 192, 0.3)',
                            backgroundColor: preview ? 'rgba(100, 200, 255, 0.1)' : 'rgba(50, 50, 50, 0.5)'
                          }}
                        >
                          {!preview && <div className="absolute inset-0 scan-line" style={{borderRadius: '0.5rem', background: 'linear-gradient(180deg, transparent, rgba(100, 200, 255, 0.2), transparent)'}}></div>}
                          {preview ? (
                            <div className="flex flex-col items-center">
                              <CheckCircle className="w-10 h-10 mb-2" style={{color: '#64c8ff'}} />
                              <span className="text-xs font-medium text-center" style={{color: '#64c8ff'}}>
                                ✓ Image Ready
                              </span>
                            </div>
                          ) : (
                            <div className="flex flex-col items-center">
                              <Upload className="w-10 h-10 mb-2" style={{color: '#808080'}} />
                              <span className="text-xs font-medium text-center" style={{color: '#b0b0b0'}}>
                                Click to upload
                              </span>
                              <span className="text-xs mt-1" style={{color: '#808080'}}>
                                PNG, JPG, JPEG
                              </span>
                            </div>
                          )}
                        </label>
                      </div>
                    </div>

                    {/* Preview */}
                    {preview && (
                      <div className="md:col-span-1">
                        <label className="block text-sm font-semibold mb-4" style={{color: '#b0b0b0'}}>
                          Preview
                        </label>
                        <img
                          src={preview}
                          alt="Preview"
                          className="w-full h-40 object-cover rounded-lg border shadow-md"
                          style={{borderColor: 'rgba(100, 200, 255, 0.3)'}}
                        />
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="md:col-span-1 flex flex-col justify-end gap-3">
                      <button
                        onClick={handleAnalyze}
                        disabled={!selectedFile || loading}
                        className={`py-3 px-4 rounded-xl font-semibold text-white flex items-center justify-center gap-2 button-hover transition-all duration-300 ${
                          !selectedFile || loading
                            ? 'bg-gray-400 cursor-not-allowed opacity-60'
                            : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg hover:shadow-xl'
                        }`}
                      >
                        {loading ? (
                          <>
                            <Loader className="w-5 h-5 animate-spin" />
                            <span>Analyzing...</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-5 h-5" />
                            <span>Analyze</span>
                          </>
                        )}
                      </button>
                      {selectedFile && !loading && (
                        <button
                          onClick={handleReset}
                          className="py-3 px-4 rounded-xl font-semibold text-gray-700 bg-gradient-to-r from-gray-200 to-gray-300 hover:from-gray-300 hover:to-gray-400 transition-all duration-300 button-hover shadow-md hover:shadow-lg"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* STEP 2: PROCESSING / RESULTS SECTION */}
                {(loading || result || error) && (
                  <div className="backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden slide-in border p-8" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.3)'}}>
                    <div className="flex items-center gap-3 mb-6">
                      <div style={{fontSize: '1.5rem', background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>📊</div>
                      <h2 className="text-2xl font-bold heading-font" style={{color: '#e8e8e8'}}>Step 2: Analysis Results</h2>
                    </div>

                    {error && (
                      <div className="flex gap-3 p-4 rounded-xl mb-4 slide-in shadow-md border" style={{backgroundColor: 'rgba(100, 50, 50, 0.5)', borderColor: 'rgba(255, 100, 100, 0.3)'}}>
                        <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{color: '#ff6b6b'}} />
                        <div>
                          <p className="font-semibold" style={{color: '#ff6b6b'}}>Error</p>
                          <p className="text-sm" style={{color: '#b0b0b0'}}>{error}</p>
                        </div>
                      </div>
                    )}

                    {loading && (
                      <div className="flex flex-col items-center justify-center py-12 slide-in">
                        <div className="relative">
                          <div className="absolute inset-0 rounded-full blur-lg opacity-50 animate-pulse" style={{background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)'}}></div>
                          <Loader className="w-12 h-12 animate-spin mb-4 relative" style={{color: '#64c8ff'}} />
                        </div>
                        <p className="font-medium mt-4" style={{color: '#b0b0b0'}}>
                          Analyzing your image...
                        </p>
                        <p className="text-sm mt-2" style={{color: '#808080'}}>
                          Processing with AI models
                        </p>
                      </div>
                    )}

                    {result && !loading && (
                      <div className="result-card-enhanced rounded-xl border p-6" style={{borderColor: 'rgba(100, 200, 255, 0.5)'}}>
                        {/* Result Header */}
                        <div className="result-header mb-4 -mx-6 -mt-6">
                          <div className="flex items-center gap-2">
                            <div style={{fontSize: '1.5rem'}}>✨</div>
                            <h3 className="text-lg font-bold heading-font" style={{color: '#64c8ff'}}>Analysis Complete</h3>
                          </div>
                        </div>

                        {/* Result Content */}
                        <div className="result-content-enhanced">
                          <div
                            className="prose prose-sm max-w-none line-clamp-none result-text"
                            style={{
                              color: '#b0b0b0',
                              fontSize: '0.95rem',
                              lineHeight: '1.6'
                            }}
                            dangerouslySetInnerHTML={{ __html: result }}
                          />
                        </div>

                        {/* Result Footer */}
                        <div className="mt-6 pt-4 border-t" style={{borderColor: 'rgba(100, 200, 255, 0.2)'}}>
                          <div className="flex items-center justify-between">
                            <span style={{color: '#808080', fontSize: '0.875rem'}}>✓ Analysis powered by AI</span>
                            <div className="flex gap-1">
                              <div className="w-2 h-2 rounded-full" style={{background: '#64c8ff'}}></div>
                              <div className="w-2 h-2 rounded-full" style={{background: '#9664ff'}}></div>
                              <div className="w-2 h-2 rounded-full" style={{background: '#64c8ff'}}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {!result && !loading && !error && (
                      <div className="flex flex-col items-center justify-center py-12 text-center slide-in">
                        <div className="mb-4" style={{color: '#808080', fontSize: '2rem'}}>📋</div>
                        <p className="font-medium" style={{color: '#b0b0b0'}}>
                          Upload and analyze an image to see results
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* STEP 3: Q&A SECTION - COOL DESIGN */}
                {result && !loading && (
                  <div className="backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden slide-in border" style={{backgroundColor: 'rgba(30, 30, 30, 0.9)', borderColor: 'rgba(100, 200, 255, 0.5)', boxShadow: '0 0 40px rgba(100, 200, 255, 0.2)'}}>
                    {/* Header with gradient background */}
                    <div className="p-8 relative overflow-hidden" style={{background: 'linear-gradient(135deg, rgba(100, 200, 255, 0.15) 0%, rgba(150, 100, 255, 0.15) 100%)', borderBottom: '2px solid rgba(100, 200, 255, 0.3)'}}>
                      <div className="absolute top-0 right-0 w-40 h-40 rounded-full blur-3xl opacity-20" style={{background: 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)'}}></div>
                      <div className="flex items-center gap-4 relative z-10">
                        <div style={{fontSize: '2.5rem', animation: 'bounce-gentle 2s ease-in-out infinite'}}>💬</div>
                        <div>
                          <h2 className="text-3xl font-bold heading-font" style={{color: '#e8e8e8', marginBottom: '0.25rem'}}>Medical AI Assistant</h2>
                          <p style={{color: '#b0b0b0', fontSize: '0.875rem'}}>Ask anything about your analysis results</p>
                        </div>
                      </div>
                    </div>

                    <div className="p-8">
                      {/* Chat Messages Container */}
                      <div className="mb-6 rounded-2xl p-6 overflow-y-auto h-64" style={{background: 'linear-gradient(135deg, rgba(20, 20, 20, 0.8) 0%, rgba(30, 30, 30, 0.8) 100%)', border: '2px solid rgba(100, 200, 255, 0.2)', boxShadow: 'inset 0 0 20px rgba(100, 200, 255, 0.05)'}}>
                        {chatMessages.length === 0 ? (
                          <div className="flex flex-col items-center justify-center h-full text-center">
                            <div style={{fontSize: '3rem', marginBottom: '1rem', opacity: 0.5}}>🤖</div>
                            <p style={{color: '#b0b0b0', fontSize: '0.95rem', marginBottom: '0.5rem'}}>No questions yet</p>
                            <p style={{color: '#808080', fontSize: '0.85rem'}}>Start by asking about treatments, symptoms, or prevention</p>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            {chatMessages.map((msg, idx) => (
                              <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}>
                                <div 
                                  className="max-w-sm px-4 py-3 rounded-2xl backdrop-blur-sm border transition-all duration-300 hover:shadow-lg"
                                  style={{
                                    background: msg.type === 'user' 
                                      ? 'linear-gradient(135deg, rgba(100, 200, 255, 0.25) 0%, rgba(100, 200, 255, 0.1) 100%)' 
                                      : msg.type === 'error'
                                      ? 'linear-gradient(135deg, rgba(255, 100, 100, 0.25) 0%, rgba(255, 100, 100, 0.1) 100%)'
                                      : 'linear-gradient(135deg, rgba(100, 255, 150, 0.25) 0%, rgba(100, 255, 150, 0.1) 100%)',
                                    borderColor: msg.type === 'user'
                                      ? 'rgba(100, 200, 255, 0.4)'
                                      : msg.type === 'error'
                                      ? 'rgba(255, 100, 100, 0.4)'
                                      : 'rgba(100, 255, 150, 0.4)',
                                    color: msg.type === 'user'
                                      ? '#64c8ff'
                                      : msg.type === 'error'
                                      ? '#ff6464'
                                      : '#64ff96',
                                    boxShadow: msg.type === 'user'
                                      ? '0 0 15px rgba(100, 200, 255, 0.2)'
                                      : msg.type === 'error'
                                      ? '0 0 15px rgba(255, 100, 100, 0.2)'
                                      : '0 0 15px rgba(100, 255, 150, 0.2)'
                                  }}
                                >
                                  <p className="text-sm leading-relaxed">{msg.text}</p>
                                </div>
                              </div>
                            ))}
                            {chatLoading && (
                              <div className="flex justify-start animate-fadeIn">
                                <div className="px-4 py-3 rounded-2xl backdrop-blur-sm border" style={{background: 'linear-gradient(135deg, rgba(100, 255, 150, 0.25) 0%, rgba(100, 255, 150, 0.1) 100%)', borderColor: 'rgba(100, 255, 150, 0.4)', color: '#64ff96'}}>
                                  <div className="flex items-center gap-2">
                                    <div className="flex gap-1">
                                      <div className="w-2 h-2 rounded-full" style={{background: '#64ff96', animation: 'pulse 1.4s infinite'}}></div>
                                      <div className="w-2 h-2 rounded-full" style={{background: '#64ff96', animation: 'pulse 1.4s infinite 0.2s'}}></div>
                                      <div className="w-2 h-2 rounded-full" style={{background: '#64ff96', animation: 'pulse 1.4s infinite 0.4s'}}></div>
                                    </div>
                                    <span className="text-sm ml-1">AI is thinking...</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Input Section */}
                      <div className="space-y-4">
                        <div className="flex gap-3">
                          <input
                            type="text"
                            value={chatInput}
                            onChange={(e) => setChatInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && !chatLoading && chatInput.trim() && handleAskMedical()}
                            placeholder="What would you like to know? Type your question..."
                            disabled={chatLoading}
                            className="flex-1 px-5 py-3 rounded-xl border focus:outline-none focus:ring-2 transition-all duration-300"
                            style={{
                              background: '#1a1a1a',
                              borderColor: chatInput ? 'rgba(100, 200, 255, 0.6)' : 'rgba(100, 200, 255, 0.2)',
                              color: '#b0b0b0',
                              boxShadow: chatInput ? '0 0 20px rgba(100, 200, 255, 0.1)' : 'none'
                            }}
                          />
                          <button
                            onClick={handleAskMedical}
                            disabled={chatLoading || !chatInput.trim()}
                            className="px-6 py-3 rounded-xl font-semibold button-hover transition-all duration-300 flex items-center gap-2"
                            style={{
                              background: chatLoading || !chatInput.trim() 
                                ? 'rgba(100, 200, 255, 0.1)' 
                                : 'linear-gradient(135deg, #64c8ff 0%, #9664ff 100%)',
                              color: '#fff',
                              opacity: chatLoading || !chatInput.trim() ? 0.5 : 1,
                              boxShadow: chatLoading || !chatInput.trim() 
                                ? 'none'
                                : '0 0 20px rgba(100, 200, 255, 0.3)',
                              cursor: chatLoading || !chatInput.trim() ? 'not-allowed' : 'pointer'
                            }}
                          >
                            {chatLoading ? (
                              <>
                                <Loader className="w-4 h-4 animate-spin" />
                                <span>Asking...</span>
                              </>
                            ) : (
                              <>
                                <span>Send</span>
                                <span style={{fontSize: '1rem'}}>✨</span>
                              </>
                            )}
                          </button>
                        </div>

                        {/* Quick suggestions */}
                        {chatMessages.length === 0 && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            <button
                              onClick={() => { setChatInput('What are the main findings in this image?'); }}
                              className="px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300 hover:shadow-lg"
                              style={{background: 'rgba(100, 200, 255, 0.1)', color: '#64c8ff', border: '1px solid rgba(100, 200, 255, 0.3)', cursor: 'pointer'}}
                            >
                              📊 Main findings?
                            </button>
                            <button
                              onClick={() => { setChatInput('What treatments are recommended?'); }}
                              className="px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300 hover:shadow-lg"
                              style={{background: 'rgba(100, 200, 255, 0.1)', color: '#64c8ff', border: '1px solid rgba(100, 200, 255, 0.3)', cursor: 'pointer'}}
                            >
                              💊 Treatments?
                            </button>
                            <button
                              onClick={() => { setChatInput('How can I prevent this condition?'); }}
                              className="px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300 hover:shadow-lg"
                              style={{background: 'rgba(100, 200, 255, 0.1)', color: '#64c8ff', border: '1px solid rgba(100, 200, 255, 0.3)', cursor: 'pointer'}}
                            >
                              🛡️ Prevention?
                            </button>
                            <button
                              onClick={() => { setChatInput('What should I do next?'); }}
                              className="px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300 hover:shadow-lg"
                              style={{background: 'rgba(100, 200, 255, 0.1)', color: '#64c8ff', border: '1px solid rgba(100, 200, 255, 0.3)', cursor: 'pointer'}}
                            >
                              ➡️ Next steps?
                            </button>
                          </div>
                        )}

                        {chatError && (
                          <div className="p-4 rounded-xl border" style={{background: 'rgba(255, 100, 100, 0.1)', borderColor: 'rgba(255, 100, 100, 0.3)', color: '#ff6464'}}>
                            <p className="text-sm flex items-center gap-2">
                              <span>⚠️</span>
                              <span>{chatError}</span>
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Footer */}
          {page === 'home' && (
            <div className="text-center mt-8 drop-shadow-md text-sm slide-in" style={{color: '#b0b0b0'}}>
              <p>Medical Imaging Analysis</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
