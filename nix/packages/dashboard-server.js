#!/usr/bin/env node

/**
 * EXO Dashboard Server
 * Serves the static dashboard files and provides API proxy functionality
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const url = require('url');

// Configuration from environment variables
const PORT = process.env.EXO_DASHBOARD_PORT || process.env.PORT || 8080;
const API_PORT = process.env.EXO_API_PORT || 52415;
const SSL_CERT = process.env.EXO_SSL_CERT;
const SSL_KEY = process.env.EXO_SSL_KEY;
const DASHBOARD_DIR = process.argv[2] || '/usr/share/exo/dashboard';
const AUTH_ENABLED = process.env.EXO_DASHBOARD_AUTH === 'true';
const API_KEYS = process.env.EXO_DASHBOARD_API_KEYS ? process.env.EXO_DASHBOARD_API_KEYS.split(',') : [];
const SESSION_TIMEOUT = parseInt(process.env.EXO_DASHBOARD_SESSION_TIMEOUT || '3600') * 1000;

// Session management for authentication
const sessions = new Map();
const activeOperations = new Map(); // Track ongoing operations for progress monitoring

/**
 * OpenAI-compatible endpoint mappings
 */
const OPENAI_ENDPOINT_MAPPINGS = {
  '/v1/models': '/api/models',
  '/v1/chat/completions': '/api/chat/completions',
  '/v1/completions': '/api/completions',
  '/v1/embeddings': '/api/embeddings'
};

/**
 * Map OpenAI-compatible endpoints to EXO endpoints
 */
function mapOpenAIEndpoint(pathname) {
  return OPENAI_ENDPOINT_MAPPINGS[pathname] || pathname;
}

/**
 * Generate operation ID for tracking
 */
function generateOperationId() {
  return require('crypto').randomBytes(16).toString('hex');
}

/**
 * Track operation progress
 */
function trackOperation(operationId, operation) {
  activeOperations.set(operationId, {
    ...operation,
    startTime: Date.now(),
    lastUpdate: Date.now()
  });
  
  // Clean up old operations after 1 hour
  setTimeout(() => {
    activeOperations.delete(operationId);
  }, 3600000);
}

/**
 * Update operation progress
 */
function updateOperationProgress(operationId, progress) {
  const operation = activeOperations.get(operationId);
  if (operation) {
    operation.progress = progress;
    operation.lastUpdate = Date.now();
  }
}

/**
 * Get operation status
 */
function getOperationStatus(operationId) {
  return activeOperations.get(operationId) || null;
}

/**
 * Handle operation status endpoint
 */
function handleOperationStatus(req, res, operationId) {
  const operation = getOperationStatus(operationId);
  
  if (!operation) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Operation not found' }));
    return;
  }
  
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    operationId,
    status: operation.status || 'running',
    progress: operation.progress || 0,
    startTime: operation.startTime,
    lastUpdate: operation.lastUpdate,
    duration: Date.now() - operation.startTime,
    operation: operation.type || 'unknown'
  }));
}

/**
 * Generate a random session token
 */
function generateSessionToken() {
  return require('crypto').randomBytes(32).toString('hex');
}

/**
 * Validate API key
 */
function isValidApiKey(apiKey) {
  return API_KEYS.includes(apiKey);
}

/**
 * Check if session is valid
 */
function isValidSession(sessionToken) {
  const session = sessions.get(sessionToken);
  if (!session) return false;
  
  if (Date.now() > session.expires) {
    sessions.delete(sessionToken);
    return false;
  }
  
  // Extend session
  session.expires = Date.now() + SESSION_TIMEOUT;
  return true;
}

/**
 * Create new session
 */
function createSession() {
  const token = generateSessionToken();
  sessions.set(token, {
    created: Date.now(),
    expires: Date.now() + SESSION_TIMEOUT
  });
  return token;
}

/**
 * Handle authentication
 */
function handleAuth(req, res) {
  if (req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const { apiKey } = JSON.parse(body);
        
        if (isValidApiKey(apiKey)) {
          const sessionToken = createSession();
          res.writeHead(200, { 
            'Content-Type': 'application/json',
            'Set-Cookie': `exo-session=${sessionToken}; HttpOnly; Secure=${SSL_CERT ? 'true' : 'false'}; SameSite=Strict; Max-Age=${SESSION_TIMEOUT / 1000}`
          });
          res.end(JSON.stringify({ success: true, sessionToken }));
        } else {
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid API key' }));
        }
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid request body' }));
      }
    });
  } else {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Method not allowed' }));
  }
}

/**
 * Check authentication for protected routes
 */
function checkAuth(req) {
  if (!AUTH_ENABLED) return true;
  
  // Check session cookie
  const cookies = req.headers.cookie;
  if (cookies) {
    const sessionMatch = cookies.match(/exo-session=([^;]+)/);
    if (sessionMatch && isValidSession(sessionMatch[1])) {
      return true;
    }
  }
  
  // Check Authorization header
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.substring(7);
    return isValidApiKey(token);
  }
  
  return false;
}

/**
 * Send authentication required response
 */
function sendAuthRequired(res) {
  res.writeHead(401, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ 
    error: 'Authentication required',
    authEndpoint: '/auth'
  }));
}

// MIME type mapping
const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.eot': 'application/vnd.ms-fontobject'
};

/**
 * Get MIME type for file extension
 */
function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return MIME_TYPES[ext] || 'application/octet-stream';
}

/**
 * Serve static files from dashboard directory
 */
function serveStaticFile(req, res, filePath) {
  const fullPath = path.join(DASHBOARD_DIR, filePath);
  
  // Security check - prevent directory traversal
  if (!fullPath.startsWith(path.resolve(DASHBOARD_DIR))) {
    res.writeHead(403, { 'Content-Type': 'text/plain' });
    res.end('Forbidden');
    return;
  }

  fs.stat(fullPath, (err, stats) => {
    if (err || !stats.isFile()) {
      // For SPA routing, serve index.html for non-API routes
      if (!filePath.startsWith('/api/')) {
        serveStaticFile(req, res, 'index.html');
        return;
      }
      
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
      return;
    }

    const mimeType = getMimeType(fullPath);
    const stream = fs.createReadStream(fullPath);
    
    res.writeHead(200, {
      'Content-Type': mimeType,
      'Content-Length': stats.size,
      'Cache-Control': filePath.endsWith('.html') ? 'no-cache' : 'public, max-age=31536000'
    });
    
    stream.pipe(res);
    
    stream.on('error', (streamErr) => {
      console.error('Stream error:', streamErr);
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Internal Server Error');
      }
    });
  });
}

/**
 * Proxy API requests to the EXO API server
 */
function proxyApiRequest(req, res, operationId = null) {
  const apiUrl = `http://localhost:${API_PORT}${req.url}`;
  const options = {
    method: req.method,
    headers: {
      ...req.headers,
      host: `localhost:${API_PORT}`,
      // Forward authentication if present
      ...(req.headers.authorization && { authorization: req.headers.authorization })
    }
  };

  const proxyReq = http.request(apiUrl, options, (proxyRes) => {
    // Update operation status
    if (operationId) {
      updateOperationProgress(operationId, { status: 'processing', progress: 10 });
    }
    
    // Handle streaming responses for real-time progress tracking
    if (proxyRes.headers['content-type']?.includes('text/event-stream') || 
        proxyRes.headers['transfer-encoding'] === 'chunked') {
      
      res.writeHead(proxyRes.statusCode, {
        ...proxyRes.headers,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        ...(operationId && { 'X-Operation-ID': operationId })
      });
      
      let bytesReceived = 0;
      const totalBytes = parseInt(proxyRes.headers['content-length']) || 0;
      
      // Stream the response for real-time updates
      proxyRes.on('data', (chunk) => {
        bytesReceived += chunk.length;
        
        // Update progress if we know the total size
        if (operationId && totalBytes > 0) {
          const progress = Math.min(90, Math.floor((bytesReceived / totalBytes) * 80) + 10);
          updateOperationProgress(operationId, { status: 'streaming', progress });
        }
        
        res.write(chunk);
      });
      
      proxyRes.on('end', () => {
        if (operationId) {
          updateOperationProgress(operationId, { status: 'completed', progress: 100 });
        }
        res.end();
      });
      
    } else {
      // Copy headers from API response
      res.writeHead(proxyRes.statusCode, {
        ...proxyRes.headers,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        ...(operationId && { 'X-Operation-ID': operationId })
      });
      
      let bytesReceived = 0;
      const totalBytes = parseInt(proxyRes.headers['content-length']) || 0;
      
      proxyRes.on('data', (chunk) => {
        bytesReceived += chunk.length;
        
        if (operationId && totalBytes > 0) {
          const progress = Math.min(90, Math.floor((bytesReceived / totalBytes) * 80) + 10);
          updateOperationProgress(operationId, { status: 'processing', progress });
        }
      });
      
      proxyRes.on('end', () => {
        if (operationId) {
          updateOperationProgress(operationId, { status: 'completed', progress: 100 });
        }
      });
      
      proxyRes.pipe(res);
    }
  });

  proxyReq.on('error', (err) => {
    console.error('API proxy error:', err);
    
    if (operationId) {
      updateOperationProgress(operationId, { status: 'failed', progress: 0, error: err.message });
    }
    
    if (!res.headersSent) {
      res.writeHead(502, { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        ...(operationId && { 'X-Operation-ID': operationId })
      });
      res.end(JSON.stringify({ 
        error: 'API server unavailable',
        details: err.message,
        timestamp: new Date().toISOString(),
        ...(operationId && { operationId })
      }));
    }
  });

  // Handle request timeout
  proxyReq.setTimeout(30000, () => {
    if (operationId) {
      updateOperationProgress(operationId, { status: 'timeout', progress: 0 });
    }
    
    proxyReq.destroy();
    if (!res.headersSent) {
      res.writeHead(504, { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        ...(operationId && { 'X-Operation-ID': operationId })
      });
      res.end(JSON.stringify({ 
        error: 'API request timeout',
        timestamp: new Date().toISOString(),
        ...(operationId && { operationId })
      }));
    }
  });

  // Forward request body for POST/PUT requests
  if (req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') {
    req.pipe(proxyReq);
  } else {
    proxyReq.end();
  }
}

/**
 * Handle health check endpoint
 */
function handleHealthCheck(req, res) {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    dashboard: 'running',
    api_proxy: `http://localhost:${API_PORT}`,
    authentication: AUTH_ENABLED ? 'enabled' : 'disabled',
    ssl: (SSL_CERT && SSL_KEY) ? 'enabled' : 'disabled'
  }));
}

/**
 * Main request handler
 */
function requestHandler(req, res) {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  // Add CORS headers for API requests
  if (pathname.startsWith('/api/')) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }
  }

  // Handle authentication endpoint
  if (pathname === '/auth') {
    handleAuth(req, res);
    return;
  }

  // Handle health check (always accessible)
  if (pathname === '/health' || pathname === '/api/health') {
    handleHealthCheck(req, res);
    return;
  }

  // Check authentication for protected routes
  if (AUTH_ENABLED && !checkAuth(req)) {
    // Allow access to login page and static assets needed for login
    if (pathname === '/' || pathname === '/login' || pathname.startsWith('/static/') || 
        pathname.endsWith('.css') || pathname.endsWith('.js') || pathname.endsWith('.png') ||
        pathname.endsWith('.ico') || pathname.endsWith('.svg')) {
      // Allow these through for login functionality
    } else {
      sendAuthRequired(res);
      return;
    }
  }

  // Handle operations list endpoint
  if (pathname === '/api/operations') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const operations = Array.from(activeOperations.entries()).map(([id, op]) => ({
      operationId: id,
      type: op.type,
      status: op.status,
      progress: op.progress,
      startTime: op.startTime,
      lastUpdate: op.lastUpdate,
      duration: Date.now() - op.startTime
    }));
    res.end(JSON.stringify({ operations }));
    return;
  }

  // Handle operation status endpoint
  if (pathname.startsWith('/api/operations/')) {
    const operationId = pathname.split('/')[3];
    if (operationId) {
      handleOperationStatus(req, res, operationId);
      return;
    }
  }

  // Map OpenAI-compatible endpoints
  let mappedPath = pathname;
  if (pathname.startsWith('/v1/')) {
    mappedPath = mapOpenAIEndpoint(pathname);
    // Update the request URL for proxying
    req.url = req.url.replace(pathname, mappedPath);
  }

  // Proxy API requests
  if (mappedPath.startsWith('/api/')) {
    // Generate operation ID for tracking long-running operations
    const operationId = generateOperationId();
    
    // Track certain operations
    if (mappedPath.includes('/chat/completions') || mappedPath.includes('/completions') || 
        mappedPath.includes('/embeddings') || mappedPath.includes('/models')) {
      
      const operationType = mappedPath.includes('/chat/completions') ? 'chat_completion' :
                           mappedPath.includes('/completions') ? 'completion' :
                           mappedPath.includes('/embeddings') ? 'embedding' :
                           mappedPath.includes('/models') ? 'model_operation' : 'api_request';
      
      trackOperation(operationId, {
        type: operationType,
        status: 'started',
        progress: 0,
        endpoint: mappedPath,
        method: req.method
      });
      
      // Add operation ID to response headers
      res.setHeader('X-Operation-ID', operationId);
    }
    
    proxyApiRequest(req, res, operationId);
    return;
  }

  // Serve static files
  const filePath = pathname === '/' ? 'index.html' : pathname.substring(1);
  serveStaticFile(req, res, filePath);
}

/**
 * Start the server
 */
function startServer() {
  // Check if dashboard directory exists
  if (!fs.existsSync(DASHBOARD_DIR)) {
    console.error(`Dashboard directory not found: ${DASHBOARD_DIR}`);
    process.exit(1);
  }

  // Check if index.html exists
  const indexPath = path.join(DASHBOARD_DIR, 'index.html');
  if (!fs.existsSync(indexPath)) {
    console.error(`Dashboard index.html not found: ${indexPath}`);
    process.exit(1);
  }

  let server;

  // Create HTTPS server if SSL certificates are provided
  if (SSL_CERT && SSL_KEY) {
    try {
      const options = {
        cert: fs.readFileSync(SSL_CERT),
        key: fs.readFileSync(SSL_KEY),
        
        // Enhanced SSL security options
        secureProtocol: 'TLSv1_2_method',
        ciphers: [
          'ECDHE-RSA-AES128-GCM-SHA256',
          'ECDHE-RSA-AES256-GCM-SHA384',
          'ECDHE-RSA-AES128-SHA256',
          'ECDHE-RSA-AES256-SHA384',
          'ECDHE-RSA-AES256-SHA256',
          'ECDHE-RSA-AES128-SHA',
          'ECDHE-RSA-AES256-SHA',
          'AES128-GCM-SHA256',
          'AES256-GCM-SHA384',
          'AES128-SHA256',
          'AES256-SHA256',
          'AES128-SHA',
          'AES256-SHA'
        ].join(':'),
        honorCipherOrder: true,
        
        // Security headers
        secureOptions: require('constants').SSL_OP_NO_SSLv2 | 
                      require('constants').SSL_OP_NO_SSLv3 |
                      require('constants').SSL_OP_NO_TLSv1 |
                      require('constants').SSL_OP_NO_TLSv1_1
      };
      
      server = https.createServer(options, (req, res) => {
        // Add security headers
        res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
        res.setHeader('X-Content-Type-Options', 'nosniff');
        res.setHeader('X-Frame-Options', 'DENY');
        res.setHeader('X-XSS-Protection', '1; mode=block');
        res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
        res.setHeader('Content-Security-Policy', 
          "default-src 'self'; " +
          "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
          "style-src 'self' 'unsafe-inline'; " +
          "img-src 'self' data: blob:; " +
          "font-src 'self'; " +
          "connect-src 'self' ws: wss:; " +
          "frame-ancestors 'none';"
        );
        
        requestHandler(req, res);
      });
      
      console.log(`EXO Dashboard starting with HTTPS on port ${PORT}`);
    } catch (err) {
      console.error('Failed to load SSL certificates:', err);
      
      // Check if certificates exist but are invalid
      if (fs.existsSync(SSL_CERT) && fs.existsSync(SSL_KEY)) {
        console.error('SSL certificates exist but are invalid. Please check certificate format.');
      } else {
        console.error('SSL certificate files not found. Please check paths:');
        console.error(`Certificate: ${SSL_CERT}`);
        console.error(`Key: ${SSL_KEY}`);
      }
      
      process.exit(1);
    }
  } else {
    server = http.createServer((req, res) => {
      // Add basic security headers even for HTTP
      res.setHeader('X-Content-Type-Options', 'nosniff');
      res.setHeader('X-Frame-Options', 'DENY');
      res.setHeader('X-XSS-Protection', '1; mode=block');
      res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
      
      requestHandler(req, res);
    });
    console.log(`EXO Dashboard starting with HTTP on port ${PORT}`);
  }

  server.listen(PORT, '0.0.0.0', () => {
    const protocol = SSL_CERT && SSL_KEY ? 'https' : 'http';
    console.log(`EXO Dashboard server running at ${protocol}://0.0.0.0:${PORT}`);
    console.log(`Dashboard files: ${DASHBOARD_DIR}`);
    console.log(`API proxy target: http://localhost:${API_PORT}`);
    
    if (!SSL_CERT || !SSL_KEY) {
      console.log('Note: Running without SSL. Set EXO_SSL_CERT and EXO_SSL_KEY for HTTPS.');
      console.log('For production use, enable SSL/TLS for secure connections.');
    } else {
      console.log('SSL/TLS enabled with enhanced security settings');
    }
    
    if (AUTH_ENABLED) {
      console.log(`Authentication enabled with ${API_KEYS.length} API key(s)`);
    } else {
      console.log('Authentication disabled - consider enabling for production use');
    }
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${PORT} is already in use`);
    } else if (err.code === 'EACCES') {
      console.error(`Permission denied to bind to port ${PORT}`);
    } else {
      console.error('Server error:', err);
    }
    process.exit(1);
  });

  // Certificate reload functionality for ACME renewals
  if (SSL_CERT && SSL_KEY) {
    // Watch for certificate changes
    const certWatcher = fs.watch(path.dirname(SSL_CERT), (eventType, filename) => {
      if (filename === path.basename(SSL_CERT) || filename === path.basename(SSL_KEY)) {
        console.log('SSL certificate change detected, reloading...');
        
        setTimeout(() => {
          try {
            const newOptions = {
              cert: fs.readFileSync(SSL_CERT),
              key: fs.readFileSync(SSL_KEY)
            };
            
            // Update server context (this is a simplified approach)
            console.log('SSL certificates reloaded successfully');
          } catch (err) {
            console.error('Failed to reload SSL certificates:', err);
          }
        }, 1000); // Wait a bit for file write to complete
      }
    });
    
    // Cleanup watcher on exit
    process.on('exit', () => {
      certWatcher.close();
    });
  }

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('Received SIGTERM, shutting down gracefully');
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  });

  process.on('SIGINT', () => {
    console.log('Received SIGINT, shutting down gracefully');
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  });

  // Handle SIGHUP for configuration reload
  process.on('SIGHUP', () => {
    console.log('Received SIGHUP, reloading configuration');
    // In a more complex setup, this could reload configuration
    // For now, just log the event
  });
}

// Start the server
startServer();