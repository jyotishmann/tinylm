//import { StrictMode } from 'react'
//import { createRoot } from 'react-dom/client'
//import './index.css'
//import App from './App.jsx'

//createRoot(document.getElementById('root')).render(
//  <StrictMode>
//    <App />
//  </StrictMode>,
//)

// src/main.jsx
// React entry point. Mounts App into #root.
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'  // Tailwind base styles

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)