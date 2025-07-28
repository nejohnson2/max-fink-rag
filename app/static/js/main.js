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
    
    this.initEventListeners();
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

      const response = await fetch('/query', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Remove typing indicator
      this.removeTypingIndicator(typingId);
      
      // Add bot response
      this.addMessage('bot', result.answer);

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

  addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    const paragraph = document.createElement('p');
    paragraph.textContent = content;
    messageContent.appendChild(paragraph);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();
    
    return messageDiv;
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