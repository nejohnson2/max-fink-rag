// Chat functionality
class ChatInterface {
  constructor() {
    this.chatMessages = document.getElementById('chatMessages');
    this.questionInput = document.getElementById('questionInput');
    this.sendBtn = document.getElementById('sendBtn');
    this.uploadForm = document.getElementById('uploadForm');
    this.queryForm = document.getElementById('queryForm');
    this.uploadStatus = document.getElementById('uploadStatus');
    this.loadingOverlay = document.getElementById('loadingOverlay');
    this.fileInput = document.getElementById('fileInput');
    this.fileLabel = document.querySelector('.file-input-label span');

    // Markdown rendering (expects `marked` to be available globally)
    this.markdown = window.marked;
    // Optional sanitization if DOMPurify is available globally
    this.sanitizer = window.DOMPurify;

    // Session management
    this.sessionId = this.initializeSession();

    this.initEventListeners();
  }

  initializeSession() {
    // Generate unique session ID for this browser tab
    // Using timestamp + random for uniqueness
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Store in sessionStorage (cleared on tab close/refresh)
    sessionStorage.setItem('chatSessionId', sessionId);

    console.log('Session initialized:', sessionId);

    // Send cleanup signal when tab closes or refreshes
    window.addEventListener('beforeunload', () => {
      // Use sendBeacon for reliable cleanup signal
      const cleanupData = new FormData();
      cleanupData.append('session_id', this.sessionId);
      navigator.sendBeacon('/cleanup_session', cleanupData);
    });

    return sessionId;
  }

  initEventListeners() {
    //Upload form handler
    if (this.uploadForm) {
      this.uploadForm.onsubmit = async (e) => {
        e.preventDefault();
        await this.handleUpload(e);
      };
    }

    if (this.queryForm) {
      this.queryForm.onsubmit = async (e) => {
        e.preventDefault();
        await this.handleQuery(e);
      };
    }

    // // File input change handler
    // this.fileInput.onchange = (e) => {
    //   const file = e.target.files[0];
    //   if (file) {
    //     this.fileLabel.textContent = file.name;
    //   } else {
    //     this.fileLabel.textContent = 'Choose PDF file';
    //   }
    // };

    // Enter key handler for chat input
    if (this.questionInput) {
      this.questionInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          document.getElementById('chatMessages').style.display = 'block';
          this.queryForm.dispatchEvent(new Event('submit'));
        }
      };
    }
  }

  showLoading() {
    this.loadingOverlay.style.display = 'flex';
  }

  hideLoading() {
    this.loadingOverlay.style.display = 'none';
  }

  showStatus(message, type = 'success') {
    this.uploadStatus.textContent = message;
    this.uploadStatus.className = `status-message ${type}`;
    this.uploadStatus.style.display = 'block';
    
    // Hide after 5 seconds
    setTimeout(() => {
      this.uploadStatus.style.display = 'none';
    }, 5000);
  }

  async handleUpload(e) {
    const formData = new FormData(e.target);
    const file = formData.get('file');
    
    if (!file || file.size === 0) {
      this.showStatus('Please select a PDF file to upload.', 'error');
      return;
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.showStatus('Please select a valid PDF file.', 'error');
      return;
    }

    try {
      this.showLoading();
      const uploadBtn = e.target.querySelector('.upload-btn');
      uploadBtn.disabled = true;
      uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
      console.log('Uploading file:', file.name);
      const response = await fetch('/upload_pdfs', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      this.showStatus(`✅ Successfully uploaded "${result.filename}" - ${result.chunks} chunks processed`, 'success');
      
      // Add a bot message to chat
      this.addMessage('bot', `Great! I've processed your PDF "${result.filename}" and it's ready for questions. You can now ask me anything about its content.`);
      
      // Reset form
      e.target.reset();
      this.fileLabel.textContent = 'Choose PDF file';

    } catch (error) {
      console.error('Upload error:', error);
      this.showStatus(`❌ Upload failed: ${error.message}`, 'error');
    } finally {
      this.hideLoading();
      const uploadBtn = e.target.querySelector('.upload-btn');
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload Document';
    }
  }

  async handleQuery(e) {
    const formData = new FormData(e.target);
    const question = formData.get('question').trim();

    if (!question) {
      return;
    }

    try {
      // Add user message to chat
      this.addMessage('user', question);

      // Clear input and disable send button
      this.questionInput.value = '';
      this.sendBtn.disabled = true;
      this.sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

      // Add typing indicator
      const typingId = this.addTypingIndicator();

      // Send JSON with session ID instead of FormData
      const response = await fetch('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          session_id: this.sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const result = await response.json();

      // Remove typing indicator
      this.removeTypingIndicator(typingId);

      // Add bot response with sources
      this.addMessage('bot', result.answer, result.sources);

    } catch (error) {
      console.error('Query error:', error);
      this.removeTypingIndicator();
      this.addMessage('bot', `Sorry, I encountered an error: ${error.message}`);
    } finally {
      // Re-enable send button
      this.sendBtn.disabled = false;
      this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
  }

  renderMarkdown(content) {
    // Fallback to plain text if a markdown parser is not available
    if (!this.markdown || typeof this.markdown.parse !== 'function') {
      const escaped = document.createElement('div');
      escaped.textContent = content;
      return escaped.innerHTML;
    }

    // Render markdown to HTML
    let html = this.markdown.parse(content, {
      gfm: true,
      breaks: true,
    });

    // Sanitize if available (recommended for any user-provided content)
    if (this.sanitizer && typeof this.sanitizer.sanitize === 'function') {
      html = this.sanitizer.sanitize(html, {
        USE_PROFILES: { html: true },
      });
    }

    return html;
  }

  postProcessMessageEl(messageContentEl) {
    // Make links open in a new tab
    const links = messageContentEl.querySelectorAll('a');
    links.forEach((a) => {
      a.setAttribute('target', '_blank');
      a.setAttribute('rel', 'noopener noreferrer');
    });
  }

  addMessage(type, content, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = this.renderMarkdown(content);
    this.postProcessMessageEl(messageContent);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    // Add sources if available
    if (sources && sources.length > 0) {
      const sourcesDiv = this.createSourcesElement(sources);
      if (sourcesDiv) {
        messageDiv.appendChild(sourcesDiv);
      }
    }

    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();

    return messageDiv;
  }

  createSourcesElement(sources) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'message-sources';

    const sourcesHeader = document.createElement('div');
    sourcesHeader.className = 'sources-header';
    sourcesHeader.innerHTML = '<i class="fas fa-book"></i> Sources';
    sourcesContainer.appendChild(sourcesHeader);

    const sourcesList = document.createElement('div');
    sourcesList.className = 'sources-list';

    // Remove duplicates based on source URL
    const uniqueSources = [];
    const seenUrls = new Set();

    sources.forEach(source => {
      // The backend returns the URL in the "source" field, not nested in metadata
      const url = source?.source;
      if (url && !seenUrls.has(url)) {
        seenUrls.add(url);
        uniqueSources.push(source);
      }
    });

    // Only create the sources section if we have valid sources
    if (uniqueSources.length === 0) {
      return null;
    }

    uniqueSources.forEach((source, index) => {
      const sourceItem = document.createElement('a');
      sourceItem.className = 'source-item';
      sourceItem.href = source.source;
      sourceItem.target = '_blank';
      sourceItem.rel = 'noopener noreferrer';

      // Extract title from the source object
      const title = source.title || source.collection || `Source ${index + 1}`;

      // Add data-title attribute for tooltip (only if title is long)
      if (title.length > 40) {
        sourceItem.setAttribute('data-title', title);
      }

      sourceItem.innerHTML = `
        <span class="source-number">${index + 1}</span>
        <span class="source-title">${this.escapeHtml(title)}</span>
        <i class="fas fa-external-link-alt"></i>
      `;

      sourcesList.appendChild(sourceItem);
    });

    sourcesContainer.appendChild(sourcesList);
    return sourcesContainer;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = '<p><i class="fas fa-ellipsis-h fa-pulse"></i> Thinking...</p>';
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(messageContent);
    
    this.chatMessages.appendChild(typingDiv);
    this.scrollToBottom();
    
    return typingDiv.id;
  }

  removeTypingIndicator(id = 'typing-indicator') {
    const typingIndicator = document.getElementById(id);
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  scrollToBottom() {
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log('Initializing chat interface');
  new ChatInterface();
});