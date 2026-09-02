/**
 * OpenVINO NPU & LM Studio Multi-Workspace RAG Studio - Frontend Controller
 * Manages dual hardware engines, project workspaces, document RAG,
 * persistent chat history, and safe conversation memorization.
 */

// Global State
const state = {
  activeTab: 'chat',              // 'chat' | 'projects' | 'history'
  activeEngine: 'openvino',        // 'openvino' | 'lmstudio'
  activeProjectId: 'default',
  activeProjectName: 'General Workspace',
  activeSessionId: `sess_${Date.now()}`,
  activeSessionTitle: 'New Chat',
  isGenerating: false,
  soundEnabled: true,
  ragEnabled: true,
  currentEventSource: null,
  modelStatus: 'unloaded',
  lmStudioConnected: false,
  messages: [],                   // [{role: 'user'|'assistant', text: '...', citations: [], timestamp: ...}]
  projects: [],
  sessions: [],
  audioContext: null
};

// DOM Elements Map
const dom = {
  // Navigation & Tabs
  tabDirectChatBtn: document.getElementById('tabDirectChatBtn'),
  tabProjectsBtn: document.getElementById('tabProjectsBtn'),
  tabHistoryBtn: document.getElementById('tabHistoryBtn'),
  viewDirectChat: document.getElementById('viewDirectChat'),
  viewProjects: document.getElementById('viewProjects'),
  viewHistory: document.getElementById('viewHistory'),
  activeWorkspaceBadge: document.getElementById('activeWorkspaceBadge'),
  hintWorkspace: document.getElementById('hintWorkspace'),

  // Engine Switcher
  engineOpenVinoBtn: document.getElementById('engineOpenVinoBtn'),
  engineLMStudioBtn: document.getElementById('engineLMStudioBtn'),
  openvinoControlGroup: document.getElementById('openvinoControlGroup'),
  lmstudioControlGroup: document.getElementById('lmstudioControlGroup'),
  deviceSelect: document.getElementById('deviceSelect'),
  modelActionBtn: document.getElementById('modelActionBtn'),
  modelStatusBadge: document.getElementById('modelStatusBadge'),
  statusText: document.getElementById('statusText'),
  loadingTimer: document.getElementById('loadingTimer'),
  lmStudioModelSelect: document.getElementById('lmStudioModelSelect'),
  refreshLMStudioBtn: document.getElementById('refreshLMStudioBtn'),
  lmStudioStatusBadge: document.getElementById('lmStudioStatusBadge'),
  lmStudioStatusText: document.getElementById('lmStudioStatusText'),

  // Active Session Info
  currentSessionTitle: document.getElementById('currentSessionTitle'),
  currentSessionWorkspace: document.getElementById('currentSessionWorkspace'),
  currentSessionEngine: document.getElementById('currentSessionEngine'),
  newDirectChatBtn: document.getElementById('newDirectChatBtn'),
  saveChatBtn: document.getElementById('saveChatBtn'),
  clearChatBtn: document.getElementById('clearChatBtn'),
  exportChatBtn: document.getElementById('exportChatBtn'),

  // Project Workspace
  projectSelector: document.getElementById('projectSelector'),
  projectDescText: document.getElementById('projectDescText'),
  createProjectModalBtn: document.getElementById('createProjectModalBtn'),
  memorizeChatBtn: document.getElementById('memorizeChatBtn'),

  // History
  historyCountBadge: document.getElementById('historyCountBadge'),
  historySearchInput: document.getElementById('historySearchInput'),
  historySessionsList: document.getElementById('historySessionsList'),

  // Documents & RAG
  ragToggle: document.getElementById('ragToggle'),
  ragStatusBadge: document.getElementById('ragStatusBadge'),
  topKSlider: document.getElementById('topKSlider'),
  topKValue: document.getElementById('topKValue'),
  fileUploadInput: document.getElementById('fileUploadInput'),
  uploadDropzone: document.getElementById('uploadDropzone'),
  attachFileBtn: document.getElementById('attachFileBtn'),
  globalDragOverlay: document.getElementById('globalDragOverlay'),
  documentsList: document.getElementById('documentsList'),
  docCount: document.getElementById('docCount'),
  clearDocsBtn: document.getElementById('clearDocsBtn'),

  // Chat Area
  chatMain: document.getElementById('chatMain'),
  messagesContainer: document.getElementById('messagesContainer'),
  welcomeScreen: document.getElementById('welcomeScreen'),
  chatForm: document.getElementById('chatForm'),
  chatInput: document.getElementById('chatInput'),
  sendBtn: document.getElementById('sendBtn'),
  streamTelemetryBar: document.getElementById('streamTelemetryBar'),
  smEngine: document.getElementById('smEngine'),
  smLiveTps: document.getElementById('smLiveTps'),
  smTtft: document.getElementById('smTtft'),
  smTokens: document.getElementById('smTokens'),
  stopGenerationBtn: document.getElementById('stopGenerationBtn'),
  citationsPreviewStrip: document.getElementById('citationsPreviewStrip'),
  citationPillsList: document.getElementById('citationPillsList'),
  citationCount: document.getElementById('citationCount'),
  closeCitationsBtn: document.getElementById('closeCitationsBtn'),

  // Telemetry & Sound
  ramMetric: document.getElementById('ramMetric'),
  cpuMetric: document.getElementById('cpuMetric'),
  lastTpsMetric: document.getElementById('lastTpsMetric'),
  soundToggleBtn: document.getElementById('soundToggleBtn'),
  soundIcon: document.getElementById('soundIcon'),
  settingsModalBtn: document.getElementById('settingsModalBtn'),
  sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
  appSidebar: document.getElementById('appSidebar'),
  toastContainer: document.getElementById('toastContainer'),

  // Modals
  newProjectModal: document.getElementById('newProjectModal'),
  closeNewProjectModal: document.getElementById('closeNewProjectModal'),
  cancelNewProjectBtn: document.getElementById('cancelNewProjectBtn'),
  confirmCreateProjectBtn: document.getElementById('confirmCreateProjectBtn'),
  newProjectName: document.getElementById('newProjectName'),
  newProjectDesc: document.getElementById('newProjectDesc'),

  memorizeModal: document.getElementById('memorizeModal'),
  closeMemorizeModal: document.getElementById('closeMemorizeModal'),
  cancelMemorizeBtn: document.getElementById('cancelMemorizeBtn'),
  confirmMemorizeBtn: document.getElementById('confirmMemorizeBtn'),
  memorizeProjectName: document.getElementById('memorizeProjectName'),
  memorizeSummaryText: document.getElementById('memorizeSummaryText'),

  settingsModal: document.getElementById('settingsModal'),
  closeSettingsModal: document.getElementById('closeSettingsModal'),
  cancelSettingsBtn: document.getElementById('cancelSettingsBtn'),
  saveSettingsBtn: document.getElementById('saveSettingsBtn'),
  settingMaxPromptLen: document.getElementById('settingMaxPromptLen'),
  settingMinResponseLen: document.getElementById('settingMinResponseLen'),
  settingGenerateHint: document.getElementById('settingGenerateHint'),

  // Generation Sliders
  temperatureSlider: document.getElementById('temperatureSlider'),
  tempValue: document.getElementById('tempValue'),
  maxTokensSlider: document.getElementById('maxTokensSlider'),
  maxTokensValue: document.getElementById('maxTokensValue')
};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
  initWorkspaceTabs();
  initEngineSwitcher();
  initProjectManager();
  initChatHistory();
  initDocumentUploads();
  initGlobalDragAndDrop();
  initChatInterface();
  initModals();
  initTelemetryPolling();
  
  // Initial data fetches
  fetchProjects();
  fetchDocuments();
  fetchDevices();
  pollEngineStatus();
});

// =============================================================================
// 1. WORKSPACE TABS (CHAT | PROJECTS | HISTORY)
// =============================================================================

function initWorkspaceTabs() {
  const tabs = [
    { btn: dom.tabDirectChatBtn, view: dom.viewDirectChat, id: 'chat' },
    { btn: dom.tabProjectsBtn, view: dom.viewProjects, id: 'projects' },
    { btn: dom.tabHistoryBtn, view: dom.viewHistory, id: 'history' }
  ];

  tabs.forEach(tab => {
    if (!tab.btn) return;
    tab.btn.addEventListener('click', () => {
      tabs.forEach(t => {
        t.btn.classList.remove('active');
        t.view.classList.remove('active');
      });
      tab.btn.classList.add('active');
      tab.view.classList.add('active');
      state.activeTab = tab.id;

      if (tab.id === 'history') {
        fetchSessions();
      } else if (tab.id === 'projects') {
        fetchProjects();
        fetchDocuments();
      }
    });
  });
}

// =============================================================================
// 2. DUAL-ENGINE SWITCHER (OPENVINO vs LM STUDIO)
// =============================================================================

function initEngineSwitcher() {
  dom.engineOpenVinoBtn?.addEventListener('click', () => switchEngine('openvino'));
  dom.engineLMStudioBtn?.addEventListener('click', () => switchEngine('lmstudio'));

  dom.refreshLMStudioBtn?.addEventListener('click', async () => {
    showToast('Probing LM Studio on port 1234...', 'info');
    await fetchLMStudioModels();
  });

  dom.lmStudioModelSelect?.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    if (!selectedModel) return;
    try {
      await fetch('/api/lmstudio/select_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
      showToast(`Selected LM Studio model: ${selectedModel}`, 'success');
      updateSessionInfoCard();
    } catch (err) {
      showToast(`Failed to select model: ${err}`, 'error');
    }
  });

  dom.modelActionBtn?.addEventListener('click', handleModelAction);
}

async function switchEngine(engine) {
  if (state.activeEngine === engine) return;
  state.activeEngine = engine;

  if (engine === 'openvino') {
    dom.engineOpenVinoBtn.classList.add('active');
    dom.engineLMStudioBtn.classList.remove('active');
    dom.openvinoControlGroup.style.display = 'flex';
    dom.lmstudioControlGroup.style.display = 'none';
  } else {
    dom.engineLMStudioBtn.classList.add('active');
    dom.engineOpenVinoBtn.classList.remove('active');
    dom.lmstudioControlGroup.style.display = 'flex';
    dom.openvinoControlGroup.style.display = 'none';
    await fetchLMStudioModels();
  }

  updateSessionInfoCard();

  try {
    await fetch('/api/engine/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine: engine })
    });
    showToast(`Switched engine to ${engine === 'openvino' ? 'OpenVINO (NPU)' : 'LM Studio'}`, 'info');
  } catch (err) {
    console.error('Failed to select engine on server:', err);
  }
}

async function fetchLMStudioModels() {
  try {
    const res = await fetch('/api/lmstudio/models');
    const data = await res.json();
    if (data.connected && data.models && data.models.length > 0) {
      dom.lmStudioModelSelect.innerHTML = '';
      data.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name;
        if (m.id === data.active_model) opt.selected = true;
        dom.lmStudioModelSelect.appendChild(opt);
      });
      dom.lmStudioStatusBadge.className = 'status-badge ready';
      dom.lmStudioStatusText.textContent = `Connected (${data.models.length})`;
      state.lmStudioConnected = true;
    } else {
      dom.lmStudioModelSelect.innerHTML = '<option value="">No models running in LM Studio</option>';
      dom.lmStudioStatusBadge.className = 'status-badge error';
      dom.lmStudioStatusText.textContent = 'Offline (Port 1234)';
      state.lmStudioConnected = false;
    }
  } catch (err) {
    dom.lmStudioStatusBadge.className = 'status-badge error';
    dom.lmStudioStatusText.textContent = 'Offline';
    state.lmStudioConnected = false;
  }
}

// =============================================================================
// 3. PROJECT WORKSPACE MANAGEMENT
// =============================================================================

function initProjectManager() {
  dom.projectSelector?.addEventListener('change', async (e) => {
    const projId = e.target.value;
    if (!projId) return;
    await selectProject(projId);
  });

  dom.createProjectModalBtn?.addEventListener('click', () => {
    dom.newProjectName.value = '';
    dom.newProjectDesc.value = '';
    dom.newProjectModal.style.display = 'flex';
    dom.newProjectName.focus();
  });

  dom.confirmCreateProjectBtn?.addEventListener('click', async () => {
    const name = dom.newProjectName.value.trim();
    const desc = dom.newProjectDesc.value.trim();
    if (!name) {
      showToast('Please enter a project name', 'error');
      return;
    }

    try {
      const res = await fetch('/api/projects/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description: desc })
      });
      const data = await res.json();
      if (data.success) {
        dom.newProjectModal.style.display = 'none';
        showToast(`Project "${name}" created!`, 'success');
        await fetchProjects();
        await selectProject(data.project.id);
      }
    } catch (err) {
      showToast(`Failed to create project: ${err}`, 'error');
    }
  });

  dom.memorizeChatBtn?.addEventListener('click', openMemorizeDialog);
}

async function fetchProjects() {
  try {
    const res = await fetch('/api/projects');
    const data = await res.json();
    state.projects = data.projects || [];
    state.activeProjectId = data.active_project_id || 'default';

    // Populate selector
    dom.projectSelector.innerHTML = '';
    state.projects.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.doc_count || 0} docs)`;
      if (p.id === state.activeProjectId) {
        opt.selected = true;
        state.activeProjectName = p.name;
        dom.projectDescText.textContent = p.description || 'No description provided.';
      }
      dom.projectSelector.appendChild(opt);
    });

    updateWorkspaceHeader();
  } catch (err) {
    console.error('Failed to fetch projects:', err);
  }
}

async function selectProject(projectId) {
  try {
    const res = await fetch('/api/projects/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId })
    });
    const data = await res.json();
    if (data.success) {
      state.activeProjectId = projectId;
      const targetProj = state.projects.find(p => p.id === projectId);
      if (targetProj) {
        state.activeProjectName = targetProj.name;
        dom.projectDescText.textContent = targetProj.description || '';
      }
      updateWorkspaceHeader();
      updateSessionInfoCard();
      await fetchDocuments();
      showToast(`Active Project: ${state.activeProjectName}`, 'info');
    }
  } catch (err) {
    showToast(`Failed to switch project: ${err}`, 'error');
  }
}

function updateWorkspaceHeader() {
  if (dom.activeWorkspaceBadge) dom.activeWorkspaceBadge.textContent = state.activeProjectName;
  if (dom.hintWorkspace) dom.hintWorkspace.textContent = state.activeProjectName;
  if (dom.memorizeProjectName) dom.memorizeProjectName.textContent = state.activeProjectName;
}

// =============================================================================
// 4. CHAT SESSIONS & PERSISTENT HISTORY
// =============================================================================

function initChatHistory() {
  dom.newDirectChatBtn?.addEventListener('click', startNewChatSession);
  
  dom.saveChatBtn?.addEventListener('click', async () => {
    if (state.messages.length === 0) {
      showToast('No messages to save.', 'info');
      return;
    }
    await autoSaveSession();
    showToast('Conversation saved to history!', 'success');
  });

  dom.clearChatBtn?.addEventListener('click', () => {
    if (confirm('Clear messages in current chat view?')) {
      state.messages = [];
      renderMessages();
      showToast('Messages cleared.', 'info');
    }
  });

  dom.exportChatBtn?.addEventListener('click', exportChatTranscript);

  dom.historySearchInput?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    renderHistoryList(query);
  });
}

function startNewChatSession() {
  // If there are existing messages, auto-save before new chat
  if (state.messages.length > 0) {
    autoSaveSession();
  }
  state.activeSessionId = `sess_${Date.now()}`;
  state.activeSessionTitle = 'New Chat';
  state.messages = [];
  renderMessages();
  updateSessionInfoCard();
  showToast('Started new conversation thread.', 'info');
}

async function autoSaveSession() {
  if (state.messages.length === 0) return;
  try {
    const res = await fetch('/api/sessions/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.activeSessionId,
        title: state.activeSessionTitle,
        messages: state.messages,
        project_id: state.activeProjectId,
        engine: state.activeEngine
      })
    });
    const data = await res.json();
    if (data.success && data.session) {
      state.activeSessionTitle = data.session.title;
      updateSessionInfoCard();
    }
  } catch (err) {
    console.error('Error auto-saving session:', err);
  }
}

async function fetchSessions() {
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    state.sessions = data.sessions || [];
    if (dom.historyCountBadge) dom.historyCountBadge.textContent = state.sessions.length;
    renderHistoryList();
  } catch (err) {
    console.error('Failed to fetch sessions:', err);
  }
}

function renderHistoryList(filterQuery = '') {
  if (!dom.historySessionsList) return;
  dom.historySessionsList.innerHTML = '';

  const filtered = state.sessions.filter(s => {
    if (!filterQuery) return true;
    return (s.title && s.title.toLowerCase().includes(filterQuery)) ||
           (s.last_snippet && s.last_snippet.toLowerCase().includes(filterQuery)) ||
           (s.project_name && s.project_name.toLowerCase().includes(filterQuery));
  });

  if (filtered.length === 0) {
    dom.historySessionsList.innerHTML = '<div class="empty-docs-msg">No matching conversations found.</div>';
    return;
  }

  filtered.forEach(sess => {
    const item = document.createElement('div');
    item.className = `history-item ${sess.id === state.activeSessionId ? 'active' : ''}`;

    const dateStr = new Date(sess.updated_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

    item.innerHTML = `
      <div class="history-header-row">
        <span class="history-title" title="${escapeHtml(sess.title)}">${escapeHtml(sess.title)}</span>
        <button class="history-delete-btn" title="Delete conversation" data-id="${sess.id}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path></svg>
        </button>
      </div>
      <div class="history-snippet">${escapeHtml(sess.last_snippet || 'No message preview')}</div>
      <div class="footer-row">
        <span class="history-proj-tag">${escapeHtml(sess.project_name || 'General')}</span>
        <span class="history-date">${dateStr}</span>
      </div>
    `;

    // Resume session on click
    item.addEventListener('click', (e) => {
      if (e.target.closest('.history-delete-btn')) return;
      loadSession(sess.id);
    });

    // Delete session
    const deleteBtn = item.querySelector('.history-delete-btn');
    deleteBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (confirm(`Delete conversation "${sess.title}"?`)) {
        await deleteSession(sess.id);
      }
    });

    dom.historySessionsList.appendChild(item);
  });
}

async function loadSession(sessionId) {
  try {
    const res = await fetch(`/api/sessions/load?id=${encodeURIComponent(sessionId)}`);
    const data = await res.json();
    if (data && data.messages) {
      state.activeSessionId = data.id;
      state.activeSessionTitle = data.title;
      state.messages = data.messages;
      
      // If session belonged to a specific project, switch to it
      if (data.project_id && data.project_id !== state.activeProjectId) {
        await selectProject(data.project_id);
      }

      renderMessages();
      updateSessionInfoCard();

      // Switch to Direct Chat tab
      dom.tabDirectChatBtn.click();
      showToast(`Resumed: "${data.title}"`, 'success');
    }
  } catch (err) {
    showToast(`Failed to load session: ${err}`, 'error');
  }
}

async function deleteSession(sessionId) {
  try {
    const res = await fetch('/api/sessions/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await res.json();
    if (data.success) {
      showToast('Conversation deleted.', 'info');
      await fetchSessions();
      if (state.activeSessionId === sessionId) {
        startNewChatSession();
      }
    }
  } catch (err) {
    showToast(`Failed to delete session: ${err}`, 'error');
  }
}

function updateSessionInfoCard() {
  if (dom.currentSessionTitle) dom.currentSessionTitle.textContent = state.activeSessionTitle;
  if (dom.currentSessionWorkspace) dom.currentSessionWorkspace.textContent = `Workspace: ${state.activeProjectName}`;
  if (dom.currentSessionEngine) {
    dom.currentSessionEngine.textContent = state.activeEngine === 'openvino' 
      ? `Engine: OpenVINO (${dom.deviceSelect.value})` 
      : `Engine: LM Studio (${dom.lmStudioModelSelect.value || 'Active'})`;
  }
}

// =============================================================================
// 5. CHAT MEMORIZATION & DISTILLATION MODAL
// =============================================================================

async function openMemorizeDialog() {
  if (state.messages.length === 0) {
    showToast('No conversation turns to distill and memorize.', 'info');
    return;
  }

  dom.memorizeProjectName.textContent = state.activeProjectName;
  dom.memorizeSummaryText.value = 'Distilling key findings and Q&A insights from conversation...';
  dom.memorizeModal.style.display = 'flex';

  try {
    const res = await fetch('/api/sessions/distill', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: state.messages,
        title: state.activeSessionTitle
      })
    });
    const data = await res.json();
    if (data.success && data.distilled_text) {
      dom.memorizeSummaryText.value = data.distilled_text;
    }
  } catch (err) {
    dom.memorizeSummaryText.value = `Failed to auto-distill: ${err}\n\nYou can manually write the notes to index here.`;
  }
}

// =============================================================================
// 6. DOCUMENTS & RAG PIPELINE
// =============================================================================

function initDocumentUploads() {
  dom.fileUploadInput?.addEventListener('change', async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    await uploadFiles(files);
    dom.fileUploadInput.value = '';
  });

  dom.ragToggle?.addEventListener('change', (e) => {
    state.ragEnabled = e.target.checked;
    if (dom.ragStatusBadge) {
      dom.ragStatusBadge.className = `rag-badge ${state.ragEnabled ? '' : 'disabled'}`;
      dom.ragStatusBadge.textContent = state.ragEnabled ? 'Active' : 'Off';
    }
    showToast(`RAG Document Mode: ${state.ragEnabled ? 'ON' : 'OFF'}`, 'info');
  });

  dom.topKSlider?.addEventListener('input', (e) => {
    dom.topKValue.textContent = e.target.value;
  });

  dom.clearDocsBtn?.addEventListener('click', async () => {
    if (confirm(`Clear all documents in "${state.activeProjectName}"?`)) {
      try {
        await fetch('/api/documents/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ project_id: state.activeProjectId })
        });
        showToast('Project knowledge base cleared.', 'info');
        await fetchDocuments();
      } catch (err) {
        showToast(`Failed to clear docs: ${err}`, 'error');
      }
    }
  });
}

function initGlobalDragAndDrop() {
  let dragCounter = 0;

  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (dom.globalDragOverlay) dom.globalDragOverlay.classList.add('active');
  });

  window.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      if (dom.globalDragOverlay) dom.globalDragOverlay.classList.remove('active');
    }
  });

  window.addEventListener('dragover', (e) => e.preventDefault());

  window.addEventListener('drop', async (e) => {
    e.preventDefault();
    dragCounter = 0;
    if (dom.globalDragOverlay) dom.globalDragOverlay.classList.remove('active');

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      await uploadFiles(files);
    }
  });
}

async function uploadFiles(files) {
  showToast(`Uploading ${files.length} document(s) to ${state.activeProjectName}...`, 'info');

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const isBinary = file.name.toLowerCase().endsWith('.pdf');

    try {
      if (isBinary) {
        const base64 = await fileToBase64(file);
        await fetch('/api/documents/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            base64: base64,
            project_id: state.activeProjectId
          })
        });
      } else {
        const text = await file.text();
        await fetch('/api/documents/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            content: text,
            project_id: state.activeProjectId
          })
        });
      }
    } catch (err) {
      showToast(`Failed to upload ${file.name}: ${err}`, 'error');
    }
  }

  showToast('Documents indexed successfully!', 'success');
  playChime('success');
  await fetchDocuments();
}

async function fetchDocuments() {
  try {
    const res = await fetch(`/api/documents?project_id=${encodeURIComponent(state.activeProjectId)}`);
    const data = await res.json();
    const docs = data.documents || [];
    if (dom.docCount) dom.docCount.textContent = docs.length;
    renderDocumentsList(docs);
  } catch (err) {
    console.error('Failed to fetch documents:', err);
  }
}

function renderDocumentsList(docs) {
  if (!dom.documentsList) return;
  dom.documentsList.innerHTML = '';

  if (docs.length === 0) {
    dom.documentsList.innerHTML = '<div class="empty-docs-msg">No documents in this project yet.</div>';
    return;
  }

  docs.forEach(doc => {
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.innerHTML = `
      <div class="doc-info-wrap">
        <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        <span class="doc-meta">${doc.chunk_count} chunks • ${doc.file_type.toUpperCase()}</span>
      </div>
      <button class="doc-delete-btn" title="Remove document" data-id="${doc.doc_id}">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>
    `;

    item.querySelector('.doc-delete-btn').addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        await fetch('/api/documents/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ doc_id: doc.doc_id, project_id: state.activeProjectId })
        });
        showToast(`Removed "${doc.filename}"`, 'info');
        await fetchDocuments();
      } catch (err) {
        showToast(`Error deleting document: ${err}`, 'error');
      }
    });

    dom.documentsList.appendChild(item);
  });
}

// =============================================================================
// 7. CHAT INTERFACE & STREAMING
// =============================================================================

function initChatInterface() {
  dom.chatForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });

  dom.chatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-grow textarea
  dom.chatInput?.addEventListener('input', () => {
    dom.chatInput.style.height = 'auto';
    dom.chatInput.style.height = `${Math.min(dom.chatInput.scrollHeight, 160)}px`;
  });

  dom.stopGenerationBtn?.addEventListener('click', () => {
    fetch('/api/chat/stop_generation', { method: 'POST' });
    if (state.currentEventSource) {
      state.currentEventSource.abort();
    }
  });

  dom.closeCitationsBtn?.addEventListener('click', () => {
    dom.citationsPreviewStrip.style.display = 'none';
  });

  // Starter Prompt Chips
  document.querySelectorAll('.starter-tag').forEach(btn => {
    btn.addEventListener('click', () => {
      const prompt = btn.getAttribute('data-prompt');
      if (prompt && dom.chatInput) {
        dom.chatInput.value = prompt;
        sendMessage();
      }
    });
  });

  // Welcome Screen Feature Cards Click Handlers
  document.getElementById('cardNPU')?.addEventListener('click', () => {
    switchEngine('openvino');
    showToast('Switched to OpenVINO (Intel NPU). Click "Start Model" if unloaded.', 'info');
  });

  document.getElementById('cardLMStudio')?.addEventListener('click', () => {
    switchEngine('lmstudio');
    showToast('Switched to LM Studio engine.', 'info');
  });

  document.getElementById('cardProjects')?.addEventListener('click', () => {
    dom.tabProjectsBtn?.click();
  });

  document.getElementById('cardHistory')?.addEventListener('click', () => {
    dom.tabHistoryBtn?.click();
  });

  // Sliders
  dom.temperatureSlider?.addEventListener('input', (e) => dom.tempValue.textContent = e.target.value);
  dom.maxTokensSlider?.addEventListener('input', (e) => dom.maxTokensValue.textContent = e.target.value);
}

async function sendMessage() {
  const text = dom.chatInput.value.trim();
  if (!text || state.isGenerating) return;

  // Check OpenVINO readiness if on OpenVINO
  if (state.activeEngine === 'openvino' && state.modelStatus !== 'ready') {
    showToast('OpenVINO model is not started. Click "Start Model" or switch to LM Studio.', 'error');
    return;
  }

  // Append User message
  const userMsg = { role: 'user', text: text, timestamp: Date.now() };
  state.messages.push(userMsg);
  dom.chatInput.value = '';
  dom.chatInput.style.height = 'auto';
  renderMessages();

  // Prepare Assistant message placeholder
  const assistantMsg = { 
    role: 'assistant', 
    text: '', 
    reasoning: '', 
    citations: [], 
    metrics: null, 
    timestamp: Date.now() 
  };
  state.messages.push(assistantMsg);
  const assistantMsgIndex = state.messages.length - 1;

  state.isGenerating = true;
  dom.sendBtn.disabled = true;
  dom.streamTelemetryBar.style.display = 'flex';
  dom.smEngine.textContent = state.activeEngine === 'openvino' ? 'OpenVINO (NPU)' : `LM Studio (${dom.lmStudioModelSelect.value || 'LLM'})`;
  dom.smLiveTps.textContent = '0.0';
  dom.smTtft.textContent = '--';
  dom.smTokens.textContent = '0';

  // Request payload
  const payload = {
    message: text,
    engine: state.activeEngine,
    project_id: state.activeProjectId,
    rag_enabled: state.ragEnabled,
    top_k: parseInt(dom.topKSlider?.value || 3),
    max_new_tokens: parseInt(dom.maxTokensSlider?.value || 512),
    temperature: parseFloat(dom.temperatureSlider?.value || 0.7),
    history: state.messages.slice(0, -2) // History without current user/assistant pair
  };

  try {
    const controller = new AbortController();
    state.currentEventSource = controller;

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Retain incomplete chunk

      let currentEvent = null;
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('event: ')) {
          currentEvent = trimmed.slice(7).trim();
        } else if (trimmed.startsWith('data: ')) {
          const dataStr = trimmed.slice(6).trim();
          if (!dataStr) continue;
          try {
            const dataObj = JSON.parse(dataStr);
            handleStreamEvent(currentEvent, dataObj, assistantMsgIndex);
          } catch (e) {
            // Ignore parse error
          }
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      state.messages[assistantMsgIndex].text += `\n\n*[Generation Error: ${err.message}]*`;
      renderMessages();
    }
  } finally {
    state.isGenerating = false;
    dom.sendBtn.disabled = false;
    dom.streamTelemetryBar.style.display = 'none';
    state.currentEventSource = null;
    autoSaveSession();
  }
}

function handleStreamEvent(event, data, msgIndex) {
  const msg = state.messages[msgIndex];
  if (!msg) return;

  if (event === 'citations') {
    msg.citations = data.citations || [];
    renderCitationsPreview(msg.citations);
    renderMessages();
  } else if (event === 'reasoning') {
    msg.reasoning = (msg.reasoning || '') + data.text;
    updateLiveMessageBubble(msgIndex);
  } else if (event === 'token') {
    msg.text += data.text;
    dom.smTokens.textContent = data.token_index || msg.text.length;
    if (data.live_tps) dom.smLiveTps.textContent = `${data.live_tps} t/s`;
    updateLiveMessageBubble(msgIndex);
  } else if (event === 'metrics') {
    msg.metrics = data;
    if (dom.lastTpsMetric) dom.lastTpsMetric.textContent = `${data.tps}`;
    dom.smTtft.textContent = `${data.ttft_ms}ms`;
    renderMessages();
    playChime('pop');
  } else if (event === 'done') {
    if (data.full_text && !msg.text) msg.text = data.full_text;
    if (data.reasoning) msg.reasoning = data.reasoning;
    renderMessages();
  } else if (event === 'error') {
    msg.text += `\n\n⚠️ **Error:** ${data.message}`;
    renderMessages();
    showToast(data.message, 'error');
  }
}

function renderMessages() {
  if (!dom.messagesContainer) return;

  if (state.messages.length === 0) {
    dom.messagesContainer.innerHTML = '';
    if (dom.welcomeScreen) dom.messagesContainer.appendChild(dom.welcomeScreen);
    return;
  }

  dom.messagesContainer.innerHTML = '';

  state.messages.forEach((msg, idx) => {
    const row = document.createElement('div');
    row.className = `message-row ${msg.role}`;
    row.id = `msg-row-${idx}`;

    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${msg.role}`;

    // Citations tags inside message
    let citationsHtml = '';
    if (msg.citations && msg.citations.length > 0) {
      citationsHtml = `
        <div class="message-citations-bar">
          <span class="cite-title">📚 Sources (${msg.citations.length}):</span>
          ${msg.citations.map(c => `
            <span class="citation-chip" title="${escapeHtml(c.preview)}">
              [${c.citation_id}] ${escapeHtml(c.filename)} (${c.score_pct}%)
            </span>
          `).join('')}
        </div>
      `;
    }

    // Reasoning block for DeepSeek / Gemma
    let reasoningHtml = '';
    if (msg.reasoning) {
      reasoningHtml = `
        <details class="reasoning-block" open>
          <summary class="reasoning-summary">💭 Thought Process</summary>
          <div class="reasoning-content">${escapeHtml(msg.reasoning)}</div>
        </details>
      `;
    }

    // Metrics tag
    let metricsHtml = '';
    if (msg.metrics) {
      metricsHtml = `
        <div class="message-metrics-tag">
          ⚡ ${msg.metrics.engine} • ${msg.metrics.tps} tokens/s • TTFT: ${msg.metrics.ttft_ms}ms • Total: ${msg.metrics.total_duration_s}s
        </div>
      `;
    }

    bubble.innerHTML = `
      ${citationsHtml}
      ${reasoningHtml}
      <div class="bubble-text">${formatMarkdown(msg.text)}</div>
      ${metricsHtml}
    `;

    row.appendChild(bubble);
    dom.messagesContainer.appendChild(row);
  });

  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
}

function updateLiveMessageBubble(msgIndex) {
  const row = document.getElementById(`msg-row-${msgIndex}`);
  if (!row) {
    renderMessages();
    return;
  }
  const msg = state.messages[msgIndex];
  const bubble = row.querySelector('.message-bubble');
  if (!bubble) return;

  let reasoningHtml = '';
  if (msg.reasoning) {
    reasoningHtml = `
      <details class="reasoning-block" open>
        <summary class="reasoning-summary">💭 Thought Process</summary>
        <div class="reasoning-content">${escapeHtml(msg.reasoning)}</div>
      </details>
    `;
  }

  bubble.innerHTML = `
    ${reasoningHtml}
    <div class="bubble-text">${formatMarkdown(msg.text)}</div>
  `;
  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
}

function renderCitationsPreview(citations) {
  if (!dom.citationsPreviewStrip) return;
  if (!citations || citations.length === 0) {
    dom.citationsPreviewStrip.style.display = 'none';
    return;
  }

  dom.citationCount.textContent = citations.length;
  dom.citationPillsList.innerHTML = '';

  citations.forEach(c => {
    const pill = document.createElement('div');
    pill.className = 'citation-pill';
    pill.innerHTML = `
      <span class="pill-id">#${c.citation_id}</span>
      <span class="pill-name">${escapeHtml(c.filename)}</span>
      <span class="pill-score">${c.score_pct}%</span>
    `;
    pill.title = c.preview;
    dom.citationPillsList.appendChild(pill);
  });

  dom.citationsPreviewStrip.style.display = 'flex';
}

function exportChatTranscript() {
  if (state.messages.length === 0) {
    showToast('No messages to export.', 'info');
    return;
  }

  let md = `# Conversation Export: ${state.activeSessionTitle}\n`;
  md += `*Workspace:* ${state.activeProjectName} | *Exported:* ${new Date().toLocaleString()}\n\n---\n\n`;

  state.messages.forEach(m => {
    const roleTitle = m.role === 'user' ? '### 👤 User' : '### 🤖 Assistant';
    md += `${roleTitle}\n${m.text}\n\n`;
    if (m.citations && m.citations.length > 0) {
      md += `*Sources Cited:*\n`;
      m.citations.forEach(c => {
        md += `- [${c.citation_id}] **${c.filename}** (Relevance: ${c.score_pct}%)\n`;
      });
      md += '\n';
    }
  });

  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${state.activeSessionTitle.toLowerCase().replace(/[^a-z0-9]/g, '_')}_export.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Chat transcript exported as Markdown!', 'success');
}

// =============================================================================
// 8. OPENVINO HARDWARE CONTROLS & POLLING
// =============================================================================

async function handleModelAction() {
  if (state.modelStatus === 'unloaded' || state.modelStatus === 'error') {
    const device = dom.deviceSelect.value;
    try {
      dom.modelStatusBadge.className = 'status-badge loading';
      dom.statusText.textContent = 'Loading...';
      dom.modelActionBtn.disabled = true;

      const res = await fetch('/api/model/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          device: device,
          config: {
            MAX_PROMPT_LEN: parseInt(dom.settingMaxPromptLen?.value || 1024),
            MIN_RESPONSE_LEN: parseInt(dom.settingMinResponseLen?.value || 512),
            GENERATE_HINT: dom.settingGenerateHint?.value || 'BEST_PERF'
          }
        })
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Compiling model on ${device}...`, 'info');
      }
    } catch (err) {
      showToast(`Error starting model: ${err}`, 'error');
    }
  } else if (state.modelStatus === 'ready') {
    try {
      await fetch('/api/model/stop', { method: 'POST' });
      showToast('OpenVINO model unloaded.', 'info');
    } catch (err) {
      showToast(`Error stopping model: ${err}`, 'error');
    }
  }
}

async function fetchDevices() {
  try {
    const res = await fetch('/api/devices');
    const data = await res.json();
    if (data.devices) {
      dom.deviceSelect.innerHTML = '';
      data.devices.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.name;
        if (d.is_recommended) opt.selected = true;
        dom.deviceSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Failed to fetch devices:', err);
  }
}

function pollEngineStatus() {
  setInterval(async () => {
    try {
      const res = await fetch('/api/engine/status');
      const data = await res.json();
      
      // Update OpenVINO status
      const ov = data.openvino;
      state.modelStatus = ov.state;

      if (ov.state === 'ready') {
        dom.modelStatusBadge.className = 'status-badge ready';
        dom.statusText.textContent = `Ready (${ov.compile_duration}s)`;
        dom.loadingTimer.style.display = 'none';
        dom.modelActionBtn.className = 'btn btn-stop';
        dom.modelActionBtn.innerHTML = '<span class="btn-text">Unload</span>';
        dom.modelActionBtn.disabled = false;
      } else if (ov.state === 'loading') {
        dom.modelStatusBadge.className = 'status-badge loading';
        dom.statusText.textContent = 'Loading';
        dom.loadingTimer.style.display = 'inline';
        dom.loadingTimer.textContent = `(${ov.elapsed_loading}s)`;
        dom.modelActionBtn.disabled = true;
      } else if (ov.state === 'error') {
        dom.modelStatusBadge.className = 'status-badge error';
        dom.statusText.textContent = 'Error';
        dom.loadingTimer.style.display = 'none';
        dom.modelActionBtn.className = 'btn btn-start';
        dom.modelActionBtn.innerHTML = '<span class="btn-text">Retry Start</span>';
        dom.modelActionBtn.disabled = false;
      } else {
        dom.modelStatusBadge.className = 'status-badge unloaded';
        dom.statusText.textContent = 'Unloaded';
        dom.loadingTimer.style.display = 'none';
        dom.modelActionBtn.className = 'btn btn-start';
        dom.modelActionBtn.innerHTML = '<span class="btn-text">Start Model</span>';
        dom.modelActionBtn.disabled = false;
      }

      // Update LM Studio status
      const lm = data.lmstudio;
      if (lm && lm.connected) {
        dom.lmStudioStatusBadge.className = 'status-badge ready';
        dom.lmStudioStatusText.textContent = `Connected (${lm.available_models ? lm.available_models.length : 0})`;
      } else if (lm) {
        dom.lmStudioStatusBadge.className = 'status-badge error';
        dom.lmStudioStatusText.textContent = 'Offline';
      }

      // System Telemetry
      if (ov.system) {
        dom.ramMetric.textContent = `${ov.system.ram_percent}%`;
        dom.cpuMetric.textContent = `${ov.system.cpu_percent}%`;
      }
    } catch (err) {
      // Ignore polling errors
    }
  }, 1000);
}

function initTelemetryPolling() {
  // Sidebar toggle
  dom.sidebarToggleBtn?.addEventListener('click', () => {
    dom.appSidebar.classList.toggle('collapsed');
  });

  // Sound toggle
  dom.soundToggleBtn?.addEventListener('click', () => {
    state.soundEnabled = !state.soundEnabled;
    dom.soundIcon.style.opacity = state.soundEnabled ? '1.0' : '0.4';
    showToast(`Audio Feedback: ${state.soundEnabled ? 'ON' : 'OFF'}`, 'info');
  });
}

// =============================================================================
// 9. MODALS & HELPERS
// =============================================================================

function initModals() {
  // New Project Modal
  dom.closeNewProjectModal?.addEventListener('click', () => dom.newProjectModal.style.display = 'none');
  dom.cancelNewProjectBtn?.addEventListener('click', () => dom.newProjectModal.style.display = 'none');

  // Memorize Modal
  dom.closeMemorizeModal?.addEventListener('click', () => dom.memorizeModal.style.display = 'none');
  dom.cancelMemorizeBtn?.addEventListener('click', () => dom.memorizeModal.style.display = 'none');
  
  dom.confirmMemorizeBtn?.addEventListener('click', async () => {
    const summary = dom.memorizeSummaryText.value.trim();
    if (!summary) {
      showToast('Summary content cannot be empty.', 'error');
      return;
    }

    try {
      const res = await fetch('/api/sessions/memorize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: state.activeProjectId,
          title: state.activeSessionTitle,
          summary_content: summary
        })
      });
      const data = await res.json();
      if (data.success) {
        dom.memorizeModal.style.display = 'none';
        showToast(`Knowledge indexed into "${state.activeProjectName}"!`, 'success');
        playChime('success');
        await fetchDocuments();
      }
    } catch (err) {
      showToast(`Failed to memorize knowledge: ${err}`, 'error');
    }
  });

  // Settings Modal
  dom.settingsModalBtn?.addEventListener('click', () => dom.settingsModal.style.display = 'flex');
  dom.closeSettingsModal?.addEventListener('click', () => dom.settingsModal.style.display = 'none');
  dom.cancelSettingsBtn?.addEventListener('click', () => dom.settingsModal.style.display = 'none');
  dom.saveSettingsBtn?.addEventListener('click', () => {
    dom.settingsModal.style.display = 'none';
    showToast('Hardware settings applied.', 'success');
  });
}

function showToast(message, type = 'info') {
  if (!dom.toastContainer) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  dom.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

function playChime(type) {
  if (!state.soundEnabled) return;
  try {
    if (!state.audioContext) {
      state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = state.audioContext;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === 'success') {
      osc.frequency.setValueAtTime(587.33, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
      osc.start();
      osc.stop(ctx.currentTime + 0.25);
    } else {
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      gain.gain.setValueAtTime(0.04, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
      osc.start();
      osc.stop(ctx.currentTime + 0.08);
    }
  } catch (e) {}
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = error => reject(error);
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

  // Code blocks ```lang ... ```
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre class="code-block"><div class="code-block-header"><span>${lang || 'code'}</span></div><code>${code.trim()}</code></pre>`;
  });

  // Line breaks
  html = html.replace(/\n/g, '<br>');

  return html;
}
