import { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = "http://localhost:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'live' | 'videos' | 'archive' | 'zones'
  const [events, setEvents] = useState([]);
  const [videos, setVideos] = useState([]);
  const [activeVideoStream, setActiveVideoStream] = useState('camera_01'); // 'camera_01' or video filename
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const fileInputRef = useRef(null);

  const [stats, setStats] = useState({
    total_events: 0,
    pending_alerts: 0,
    verified_events: 0,
    dismissed_events: 0,
    events_by_type: {},
    active_cameras: 1
  });
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [loiterThreshold, setLoiterThreshold] = useState(15);
  const [testTriggerLoading, setTestTriggerLoading] = useState(false);

  // ── Real-time alert toast state ──
  const [latestAlert, setLatestAlert] = useState(null);
  const [showToast, setShowToast] = useState(false);
  const lastAlertTsRef = useRef(null);

  // Load events, stats, and videos from FastAPI backend
  async function fetchData() {
    try {
      const [eventsRes, statsRes, videosRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/api/events?limit=100`),
        fetch(`${API_BASE}/api/events/stats/summary`),
        fetch(`${API_BASE}/api/videos`),
        fetch(`${API_BASE}/health`)
      ]);

      if (eventsRes.ok && statsRes.ok) {
        const eventsData = await eventsRes.json();
        const statsData = await statsRes.json();
        setEvents(eventsData.events || []);
        setStats(statsData);
        setIsBackendOnline(true);
      } else {
        setIsBackendOnline(false);
      }

      if (videosRes.ok) {
        const vids = await videosRes.json();
        setVideos(vids || []);
      }
    } catch (err) {
      setIsBackendOnline(false);
    }
  }

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, 3000);
    return () => clearInterval(timer);
  }, []);

  // ── Poll /api/alerts/latest for real-time toast notifications ──
  useEffect(() => {
    let cancelled = false;
    const pollAlert = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/alerts/latest`);
        if (!res.ok || res.status === 204) return;
        const data = await res.json();
        const ts = data.alerted_at || data.timestamp;
        if (ts && ts !== lastAlertTsRef.current) {
          lastAlertTsRef.current = ts;
          if (!cancelled) {
            setLatestAlert(data);
            setShowToast(true);
            // Auto-dismiss toast after 6 seconds
            setTimeout(() => { if (!cancelled) setShowToast(false); }, 6000);
          }
        }
      } catch (_) { /* backend may be offline */ }
    };
    pollAlert();
    const alertTimer = setInterval(pollAlert, 3000);
    return () => { cancelled = true; clearInterval(alertTimer); };
  }, []);

  // Update Event Review Status (Verified / Dismissed / Resolved)
  async function updateStatus(eventId, newStatus) {
    try {
      const res = await fetch(`${API_BASE}/api/events/${eventId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus, notes: `Reviewed via Web Dashboard as ${newStatus}` })
      });
      if (res.ok) {
        fetchData();
        if (selectedEvent && selectedEvent.id === eventId) {
          setSelectedEvent(prev => ({ ...prev, status: newStatus }));
        }
      }
    } catch (err) {
      console.error("Failed to update event status:", err);
    }
  }

  // Handle Video Upload
  async function handleVideoUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadMessage('Uploading testcase video to backend...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/videos/upload`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        setUploadMessage(`Success! Uploaded ${data.video?.filename}`);
        fetchData();
        setTimeout(() => setUploadMessage(''), 4000);
      } else {
        const errData = await res.json();
        setUploadMessage(`Upload failed: ${errData.detail || 'Unknown error'}`);
      }
    } catch (err) {
      setUploadMessage(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  // Delete Video
  async function deleteVideo(filename) {
    if (!confirm(`Are you sure you want to delete '${filename}'?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/videos/${filename}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        if (activeVideoStream === filename) {
          setActiveVideoStream('camera_01');
        }
        fetchData();
      }
    } catch (err) {
      console.error(err);
    }
  }

  // Trigger manual synthetic event for quick demonstration
  async function triggerSimulatedAlert(type) {
    setTestTriggerLoading(true);
    try {
      await fetch(`${API_BASE}/api/events`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: activeVideoStream === 'camera_01' ? 'camera_01' : activeVideoStream,
          activity_type: type,
          timestamp: new Date().toISOString(),
          track_id: Math.floor(Math.random() * 90) + 10,
          confidence: 0.94,
          snapshot_path: "evidence/sample_demo.jpg",
          details: {
            simulation: true,
            note: `Manual demonstration event of ${type} triggered from dashboard`
          },
          status: "pending_review"
        })
      });
      fetchData();
    } catch (err) {
      console.error(err);
    } finally {
      setTestTriggerLoading(false);
    }
  }

  // Filtered events
  const filteredEvents = events.filter(ev => {
    if (filterType !== 'all' && ev.activity_type !== filterType) return false;
    if (filterStatus !== 'all' && ev.status !== filterStatus) return false;
    return true;
  });

  // Determine current stream URL
  const currentStreamUrl = activeVideoStream === 'camera_01'
    ? `${API_BASE}/api/stream/camera_01`
    : `${API_BASE}/api/videos/${activeVideoStream}/stream`;

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="header">
        <div className="brand-section">
          <div className="brand-icon">🛡️</div>
          <div>
            <h1 className="brand-title">AI Suspicious Activity Detection</h1>
            <p className="brand-subtitle">Real-Time CCTV Surveillance & Behaviour Analysis • YOLOv11 + ByteTrack</p>
          </div>
        </div>

        <div className="header-status-pills">
          <div className="status-pill">
            <span className={`status-dot ${isBackendOnline ? 'active' : ''}`} style={{ background: isBackendOnline ? '#238636' : '#da3633' }}></span>
            <span>API: {isBackendOnline ? 'Connected' : 'Offline'}</span>
          </div>
          <div className="status-pill">
            <span>📹 Source: {activeVideoStream === 'camera_01' ? 'Live Camera (CAM-01)' : `Video: ${activeVideoStream}`}</span>
          </div>
          <div className="status-pill">
            <span>⚡ AI Model: YOLO11n</span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Overview & Metrics
        </button>
        <button
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          📹 Live Surveillance Feed
        </button>
        <button
          className={`tab-btn ${activeTab === 'videos' ? 'active' : ''}`}
          onClick={() => setActiveTab('videos')}
        >
          🎬 Video Test Cases ({videos.length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'archive' ? 'active' : ''}`}
          onClick={() => setActiveTab('archive')}
        >
          🗂️ Incident Evidence Archive ({events.length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'zones' ? 'active' : ''}`}
          onClick={() => setActiveTab('zones')}
        >
          ⚙️ Zone & Rules Configuration
        </button>
      </nav>

      {/* Metrics Row */}
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-title">Total Incidents</span>
          <span className="stat-val blue">{stats.total_events}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">Pending Security Review</span>
          <span className="stat-val red">{stats.pending_alerts}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">Restricted Trespassing</span>
          <span className="stat-val red">{stats.events_by_type?.restricted_zone_entry || 0}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">Loitering Incidents</span>
          <span className="stat-val amber">{stats.events_by_type?.loitering || 0}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">Unattended Luggage</span>
          <span className="stat-val blue">{stats.events_by_type?.abandoned_object || 0}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">Verified / Resolved</span>
          <span className="stat-val green">{stats.verified_events}</span>
        </div>
      </div>

      {/* TAB 1: OVERVIEW */}
      {activeTab === 'overview' && (
        <section>
          <div className="table-panel">
            <div className="table-header">
              <h3>🚨 Priority Incident Feed</h3>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  className="btn btn-secondary"
                  disabled={testTriggerLoading}
                  onClick={() => triggerSimulatedAlert('restricted_zone_entry')}
                >
                  + Simulate Trespassing Drill
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={testTriggerLoading}
                  onClick={() => triggerSimulatedAlert('loitering')}
                >
                  + Simulate Loitering Drill
                </button>
                <button className="btn btn-secondary" onClick={fetchData}>
                  🔄 Refresh
                </button>
              </div>
            </div>

            {events.length === 0 ? (
              <div style={{ padding: '36px', textAlign: 'center', color: '#8b949e' }}>
                <p>No suspicious events logged yet. The surveillance pipeline is standing by.</p>
              </div>
            ) : (
              <table className="events-table">
                <thead>
                  <tr>
                    <th>Evidence</th>
                    <th>Activity Type</th>
                    <th>Camera / Source</th>
                    <th>Track ID</th>
                    <th>Timestamp</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {events.slice(0, 8).map((ev) => (
                    <tr key={ev.id || ev._id}>
                      <td>
                        {ev.snapshot_url ? (
                          <img
                            src={`${API_BASE}${ev.snapshot_url}`}
                            alt="Evidence Snapshot"
                            className="thumbnail-preview"
                            onClick={() => setSelectedEvent(ev)}
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        ) : (
                          <span style={{ fontSize: '0.75rem', color: '#8b949e' }}>No Snap</span>
                        )}
                      </td>
                      <td>
                        <span className={`badge badge-${ev.activity_type?.split('_')[0] || 'status'}`}>
                          {ev.activity_type?.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td>{ev.camera_id}</td>
                      <td>#{ev.track_id ?? 'N/A'}</td>
                      <td>{new Date(ev.timestamp).toLocaleTimeString()}</td>
                      <td>
                        <span className={`badge badge-status ${ev.status}`}>
                          {ev.status}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          <button
                            className="btn btn-secondary"
                            style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                            onClick={() => setSelectedEvent(ev)}
                          >
                            Inspect
                          </button>
                          {ev.status === 'pending_review' && (
                            <>
                              <button
                                className="btn btn-primary"
                                style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                                onClick={() => updateStatus(ev.id, 'verified')}
                              >
                                Verify
                              </button>
                              <button
                                className="btn btn-danger"
                                style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                                onClick={() => updateStatus(ev.id, 'dismissed')}
                              >
                                Dismiss
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      )}

      {/* TAB 2: LIVE SURVEILLANCE FEED */}
      {activeTab === 'live' && (
        <section className="live-stream-panel">
          <div className="video-container">
            <div className="video-header-hud">
              <span className="hud-pill">
                🔴 SOURCE: {activeVideoStream === 'camera_01' ? 'LIVE FEED (CAM-01)' : `TESTCASE (${activeVideoStream})`}
              </span>
              <span className="hud-pill">FPS: ~25.0</span>
            </div>

            {isStreaming ? (
              <img
                src={currentStreamUrl}
                alt="Live Surveillance Stream"
                className="live-feed-img"
                onError={(e) => {
                  console.warn("Stream buffer connecting...");
                }}
              />
            ) : (
              <div style={{ height: '480px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#090d12' }}>
                <p style={{ color: '#8b949e' }}>Stream Paused</p>
              </div>
            )}

            <div className="video-controls">
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setIsStreaming(!isStreaming)}
                >
                  {isStreaming ? '⏸ Pause Stream' : '▶ Resume Stream'}
                </button>
                <select
                  value={activeVideoStream}
                  onChange={(e) => setActiveVideoStream(e.target.value)}
                  style={{ background: '#21262d', border: '1px solid #30363d', color: '#fff', padding: '6px 10px', borderRadius: '6px' }}
                >
                  <option value="camera_01">📹 Live Camera (CAM-01 / Webcam)</option>
                  {videos.map(v => (
                    <option key={v.filename} value={v.filename}>
                      🎬 Video: {v.filename} ({v.duration_sec}s)
                    </option>
                  ))}
                </select>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    const img = document.querySelector('.live-feed-img');
                    if (img) img.src = `${currentStreamUrl}?t=${Date.now()}`;
                  }}
                >
                  🔄 Reconnect
                </button>
              </div>
              <span style={{ fontSize: '0.8rem', color: '#8b949e' }}>
                Zones Rendered: Zone A (Restricted) & Zone B (Loitering)
              </span>
            </div>
          </div>

          {/* Telemetry Sidebar */}
          <aside className="telemetry-sidebar">
            <h4 className="sidebar-title">Feed Telemetry & Status</h4>
            <div className="telemetry-item">
              <span className="telemetry-label">Active Source</span>
              <span className="telemetry-value" style={{ maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {activeVideoStream === 'camera_01' ? 'CAM-01 (Webcam/RTSP)' : activeVideoStream}
              </span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Tracker Algorithm</span>
              <span className="telemetry-value">ByteTrack Multi-Object</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Target Classes</span>
              <span className="telemetry-value">Person, Bags, Vehicles</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Restricted Debounce</span>
              <span className="telemetry-value">5 Consecutive Frames</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Loiter Threshold</span>
              <span className="telemetry-value">{loiterThreshold} seconds</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Alert Cooldown</span>
              <span className="telemetry-value">45 seconds / ID</span>
            </div>

            <div style={{ marginTop: 'auto', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setActiveTab('videos')}
                style={{ width: '100%', justifyContent: 'center' }}
              >
                📁 Manage / Upload Video Testcases
              </button>
              <button
                className="btn btn-danger"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={() => triggerSimulatedAlert('restricted_zone_entry')}
              >
                🚨 Trigger Security Alert Test
              </button>
            </div>
          </aside>
        </section>
      )}

      {/* TAB 3: VIDEO TEST CASES & UPLOAD */}
      {activeTab === 'videos' && (
        <section className="table-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <h3>🎬 Video Test Cases & Evaluation</h3>
              <p style={{ color: '#8b949e', fontSize: '0.88rem' }}>
                Upload your own recorded CCTV or test footage (.mp4, .avi, .mov) to evaluate YOLOv11 and behaviour analysis rules.
              </p>
            </div>
            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              {isUploading ? 'Uploading...' : '➕ Upload New Test Video'}
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleVideoUpload}
              accept=".mp4,.avi,.mov,.mkv"
              style={{ display: 'none' }}
            />
          </div>

          {uploadMessage && (
            <div style={{ padding: '12px 16px', background: '#1f6feb22', border: '1px solid #1f6feb', borderRadius: '8px', marginBottom: '20px', color: '#79c0ff' }}>
              {uploadMessage}
            </div>
          )}

          {/* Upload Dropzone */}
          <div
            className="upload-dropzone"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">📹</div>
            <h4 style={{ color: '#e6edf3', marginBottom: '4px' }}>Drop your video file here or browse</h4>
            <p style={{ color: '#8b949e', fontSize: '0.85rem' }}>
              Supports MP4, AVI, MOV, MKV files. Videos are saved to <code>data/test_videos/</code>
            </p>
          </div>

          {/* Video Library Grid */}
          <h4 style={{ marginBottom: '12px' }}>Available Testcase Videos ({videos.length})</h4>
          {videos.length === 0 ? (
            <p style={{ color: '#8b949e' }}>No test videos available. Upload a video above to test the AI detection pipeline.</p>
          ) : (
            <div className="video-grid">
              {videos.map(v => (
                <div key={v.filename} className="video-card">
                  <div className="video-card-header">
                    <span className="video-filename" title={v.filename}>🎬 {v.filename}</span>
                    <span className="badge badge-status">{v.size_mb} MB</span>
                  </div>

                  <div className="video-meta-row">
                    <div className="video-meta-item">Duration: <span>{v.duration_sec}s</span></div>
                    <div className="video-meta-item">FPS: <span>{v.fps}</span></div>
                    <div className="video-meta-item">Resolution: <span>{v.width}x{v.height}</span></div>
                    <div className="video-meta-item">Frames: <span>{v.frame_count}</span></div>
                  </div>

                  <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                    <button
                      className="btn btn-primary"
                      style={{ flex: 1, justifyContent: 'center', fontSize: '0.8rem' }}
                      onClick={() => {
                        setActiveVideoStream(v.filename);
                        setActiveTab('live');
                      }}
                    >
                      ▶ Play in Live Feed
                    </button>
                    <button
                      className="btn btn-secondary"
                      style={{ fontSize: '0.8rem' }}
                      onClick={() => {
                        navigator.clipboard.writeText(`.\\ai-engine\\.venv\\Scripts\\python.exe tests\\run_video_testcase.py --video "${v.filename}"`);
                        alert(`Copied CLI run command to clipboard:\n.\\ai-engine\\.venv\\Scripts\\python.exe tests\\run_video_testcase.py --video "${v.filename}"`);
                      }}
                    >
                      📋 Copy CLI Cmd
                    </button>
                    <button
                      className="btn btn-danger"
                      style={{ fontSize: '0.8rem', padding: '4px 8px' }}
                      onClick={() => deleteVideo(v.filename)}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Instructions Box */}
          <div style={{ marginTop: '28px', background: '#0d1117', padding: '18px', borderRadius: '10px', border: '1px solid #30363d' }}>
            <h4 style={{ color: '#58a6ff', marginBottom: '8px' }}>💡 How to run batch evaluation on your video:</h4>
            <p style={{ color: '#8b949e', fontSize: '0.85rem', marginBottom: '8px' }}>
              You can execute automated testcase analysis on any uploaded video from the terminal with full report generation:
            </p>
            <div className="cli-code-block">
              .\ai-engine\.venv\Scripts\python.exe tests\run_video_testcase.py --video "sample_walkthrough.mp4" --loiter-threshold 10
            </div>
          </div>
        </section>
      )}

      {/* TAB 4: INCIDENT EVIDENCE ARCHIVE */}
      {activeTab === 'archive' && (
        <section className="table-panel">
          <div className="table-header">
            <h3>Incident History & Forensic Evidence</h3>
            <div className="filter-group">
              <label style={{ fontSize: '0.8rem', color: '#8b949e' }}>Type:</label>
              <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
                <option value="all">All Types</option>
                <option value="restricted_zone_entry">Restricted Zone Entry</option>
                <option value="loitering">Loitering</option>
                <option value="abandoned_object">Abandoned Object</option>
              </select>

              <label style={{ fontSize: '0.8rem', color: '#8b949e' }}>Status:</label>
              <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
                <option value="all">All Statuses</option>
                <option value="pending_review">Pending Review</option>
                <option value="verified">Verified</option>
                <option value="dismissed">Dismissed</option>
                <option value="resolved">Resolved</option>
              </select>
            </div>
          </div>

          <table className="events-table">
            <thead>
              <tr>
                <th>Snapshot</th>
                <th>Incident ID</th>
                <th>Activity</th>
                <th>Source</th>
                <th>Track ID</th>
                <th>Confidence</th>
                <th>Timestamp</th>
                <th>Review Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredEvents.map((ev) => (
                <tr key={ev.id || ev._id}>
                  <td>
                    {ev.snapshot_url ? (
                      <img
                        src={`${API_BASE}${ev.snapshot_url}`}
                        alt="Evidence"
                        className="thumbnail-preview"
                        onClick={() => setSelectedEvent(ev)}
                      />
                    ) : (
                      <span style={{ fontSize: '0.75rem', color: '#8b949e' }}>N/A</span>
                    )}
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                    {ev.id?.substring(0, 8) || 'evt_01'}...
                  </td>
                  <td>
                    <span className={`badge badge-${ev.activity_type?.split('_')[0] || 'status'}`}>
                      {ev.activity_type?.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td>{ev.camera_id}</td>
                  <td>#{ev.track_id ?? 'N/A'}</td>
                  <td>{Math.round((ev.confidence || 0.85) * 100)}%</td>
                  <td>{new Date(ev.timestamp).toLocaleString()}</td>
                  <td>
                    <span className={`badge badge-status ${ev.status}`}>
                      {ev.status}
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                      onClick={() => setSelectedEvent(ev)}
                    >
                      Forensic View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {/* TAB 5: ZONE CONFIGURATION */}
      {activeTab === 'zones' && (
        <section className="table-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px' }}>Zone Polygon & Detection Parameters</h3>
          <p style={{ color: '#8b949e', marginBottom: '24px', fontSize: '0.9rem' }}>
            Configure spatial polygons and dwell timers applied by the behaviour engine.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
            <div style={{ background: '#0d1117', padding: '18px', borderRadius: '8px', border: '1px solid #30363d' }}>
              <h4 style={{ color: '#ff7b72', marginBottom: '12px' }}>Zone A: Restricted Area (Red)</h4>
              <p style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '12px' }}>
                Any person track ground contact point inside this polygon triggers an immediate trespassing alarm.
              </p>
              <div style={{ fontSize: '0.85rem', marginBottom: '8px' }}>
                <strong>Vertices (x, y):</strong> (120, 100), (520, 100), (520, 400), (120, 400)
              </div>
              <div style={{ fontSize: '0.85rem' }}>
                <strong>Intrusion Debounce:</strong> 5 consecutive frames
              </div>
            </div>

            <div style={{ background: '#0d1117', padding: '18px', borderRadius: '8px', border: '1px solid #30363d' }}>
              <h4 style={{ color: '#d29922', marginBottom: '12px' }}>Zone B: Loitering Watch (Yellow)</h4>
              <p style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '12px' }}>
                Tracks remaining inside this perimeter longer than the dwell duration are flagged as loitering.
              </p>
              <div style={{ marginBottom: '12px' }}>
                <label style={{ fontSize: '0.85rem', display: 'block', marginBottom: '6px' }}>
                  Loitering Dwell Threshold (Seconds):
                </label>
                <input
                  type="number"
                  min="5"
                  max="120"
                  value={loiterThreshold}
                  onChange={(e) => setLoiterThreshold(Number(e.target.value))}
                  style={{ width: '100px' }}
                />
              </div>
              <div style={{ fontSize: '0.85rem' }}>
                <strong>Alert Cooldown:</strong> 45 seconds per Track ID
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Forensic Evidence Modal */}
      {selectedEvent && (
        <div className="modal-overlay" onClick={() => setSelectedEvent(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Forensic Evidence Inspection</h3>
              <button
                className="btn btn-secondary"
                style={{ padding: '2px 8px' }}
                onClick={() => setSelectedEvent(null)}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              {selectedEvent.snapshot_url ? (
                <img
                  src={`${API_BASE}${selectedEvent.snapshot_url}`}
                  alt="Full Evidence Snapshot"
                  className="modal-img"
                />
              ) : (
                <div style={{ height: '240px', background: '#000', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <p style={{ color: '#8b949e' }}>Snapshot file: {selectedEvent.snapshot_path || 'No snapshot saved'}</p>
                </div>
              )}

              <div className="modal-details-grid">
                <div>
                  <strong>Incident ID:</strong> {selectedEvent.id}
                </div>
                <div>
                  <strong>Activity Type:</strong>{' '}
                  <span className={`badge badge-${selectedEvent.activity_type?.split('_')[0]}`}>
                    {selectedEvent.activity_type}
                  </span>
                </div>
                <div>
                  <strong>Camera ID:</strong> {selectedEvent.camera_id}
                </div>
                <div>
                  <strong>Track ID:</strong> #{selectedEvent.track_id}
                </div>
                <div>
                  <strong>Detection Confidence:</strong> {Math.round((selectedEvent.confidence || 0.85) * 100)}%
                </div>
                <div>
                  <strong>Timestamp:</strong> {new Date(selectedEvent.timestamp).toLocaleString()}
                </div>
              </div>

              {selectedEvent.details && (
                <div style={{ background: '#0d1117', padding: '12px', borderRadius: '6px', fontSize: '0.85rem' }}>
                  <strong>Rule Telemetry Details:</strong>
                  <pre style={{ marginTop: '6px', color: '#58a6ff' }}>
                    {JSON.stringify(selectedEvent.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="btn btn-danger"
                onClick={() => {
                  updateStatus(selectedEvent.id, 'dismissed');
                  setSelectedEvent(null);
                }}
              >
                Dismiss False Positive
              </button>
              <button
                className="btn btn-primary"
                onClick={() => {
                  updateStatus(selectedEvent.id, 'verified');
                  setSelectedEvent(null);
                }}
              >
                Confirm Verified Incident
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setSelectedEvent(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Real-time Alert Toast ── */}
      {showToast && latestAlert && (
        <div className="alert-toast">
          <span style={{ fontSize: '1.3rem' }}>🚨</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '2px' }}>
              ALERT: {latestAlert.activity_type?.replace(/_/g, ' ').toUpperCase()}
            </div>
            <div style={{ fontSize: '0.8rem', opacity: 0.9 }}>
              Confidence: {Math.round((latestAlert.confidence || 0.85) * 100)}%
              {' • '}Camera: {latestAlert.camera_id || 'N/A'}
              {' • '}{new Date(latestAlert.timestamp).toLocaleTimeString()}
            </div>
          </div>
          <button
            onClick={() => setShowToast(false)}
            style={{
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.4)',
              color: '#fff',
              borderRadius: '4px',
              padding: '2px 8px',
              cursor: 'pointer',
              fontSize: '0.8rem'
            }}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
