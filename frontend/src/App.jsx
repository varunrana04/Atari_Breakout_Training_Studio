import { useState } from 'react';
import LandingPage from './pages/LandingPage.jsx';
import PlayerGame from './pages/PlayerGame.jsx';
import Studio from './pages/Studio.jsx';
import VersusMode from './pages/VersusMode.jsx';
import VersusSetupModal from './components/VersusSetupModal.jsx';
import './index.css';

export default function App() {
  // 'landing' | 'player' | 'studio' | 'versus'
  const [screen, setScreen] = useState('landing');
  const [settings, setSettings] = useState({ powerUpsEnabled: true });
  const [versusConfig, setVersusConfig] = useState(null);
  const [showVersusModal, setShowVersusModal] = useState(false);
  const [checkpoints, setCheckpoints] = useState([]);

  const openVersusModal = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/checkpoints');
      const data = await res.json();
      setCheckpoints(data.checkpoints ?? []);
    } catch (_) {
      setCheckpoints([]);
    }
    setShowVersusModal(true);
  };

  const startVersus = (config) => {
    setVersusConfig(config);
    setShowVersusModal(false);
    setScreen('versus');
  };

  return (
    <div className="app-root">
      {screen === 'landing' && (
        <LandingPage
          onPlayPlayer={() => setScreen('player')}
          onOpenStudio={() => setScreen('studio')}
          onVersus={openVersusModal}
          settings={settings}
          onSettingsChange={setSettings}
        />
      )}
      {screen === 'player' && (
        <PlayerGame
          settings={settings}
          onExit={() => setScreen('landing')}
        />
      )}
      {screen === 'studio' && (
        <Studio
          settings={settings}
          onExit={() => setScreen('landing')}
          onVersus={openVersusModal}
        />
      )}
      {screen === 'versus' && versusConfig && (
        <VersusMode
          mode={versusConfig.mode}
          agent1Episode={versusConfig.agent1Episode}
          agent2Episode={versusConfig.agent2Episode}
          onExit={() => setScreen('landing')}
        />
      )}
      {showVersusModal && (
        <VersusSetupModal
          checkpoints={checkpoints}
          onStart={startVersus}
          onClose={() => setShowVersusModal(false)}
        />
      )}
    </div>
  );
}
