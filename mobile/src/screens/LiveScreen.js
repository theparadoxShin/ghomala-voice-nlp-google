/**
 * LiveScreen — NAM SA'
 * Real-time voice conversation with Gemini via WebSocket.
 * Push-to-talk: hold mic → speak → release → agent responds with voice + text.
 * Uses ADK on backend with Gemini multimodal (audio understanding) + Cloud TTS.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, FlatList,
  Animated, Dimensions, Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAudioRecorder, RecordingPresets, requestRecordingPermissionsAsync } from 'expo-audio';
import { useLanguage } from '../context/LanguageContext';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '../theme';
import Constants from 'expo-constants';

const PROD_URL = 'https://nam-sa-976647416990.us-central1.run.app';
const API_BASE = __DEV__
  ? (Constants.expoConfig?.extra?.API_URL || 'http://192.168.1.100:8080')
  : (Constants.expoConfig?.extra?.API_URL || PROD_URL);
const WS_BASE = API_BASE.replace(/^http/, 'ws');

// Voice-optimized recording options
const VOICE_RECORDING_OPTIONS = {
  extension: '.m4a',
  sampleRate: 16000,
  numberOfChannels: 1,
  bitRate: 64000,
  android: {
    outputFormat: 'mpeg4',
    audioEncoder: 'aac',
  },
  ios: {
    outputFormat: 'aac',
    audioQuality: 'high',
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: {
    mimeType: 'audio/webm',
    bitsPerSecond: 64000,
  },
};

const STATES = {
  IDLE: 'idle',
  CONNECTING: 'connecting',
  READY: 'ready',
  RECORDING: 'recording',
  PROCESSING: 'processing',
  RESPONDING: 'responding',
};

const { width: SCREEN_W } = Dimensions.get('window');

export default function LiveScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const { t, lang } = useLanguage();

  const [status, setStatus] = useState(STATES.IDLE);
  const [messages, setMessages] = useState([]);

  const wsRef = useRef(null);
  const playerRef = useRef(null);
  const flatListRef = useRef(null);

  // Audio recorder hook
  const recorder = useAudioRecorder(VOICE_RECORDING_OPTIONS);

  // Animations
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const ringAnim = useRef(new Animated.Value(0)).current;

  // Pulse during recording
  useEffect(() => {
    if (status === STATES.RECORDING) {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.15, duration: 500, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
        ])
      );
      const ring = Animated.loop(
        Animated.sequence([
          Animated.timing(ringAnim, { toValue: 1, duration: 1200, useNativeDriver: true }),
          Animated.timing(ringAnim, { toValue: 0, duration: 0, useNativeDriver: true }),
        ])
      );
      pulse.start();
      ring.start();
      return () => { pulse.stop(); ring.stop(); };
    } else {
      pulseAnim.setValue(1);
      ringAnim.setValue(0);
    }
  }, [status]);

  // Connect WebSocket on mount
  useEffect(() => {
    connectWebSocket();
    return () => {
      wsRef.current?.close();
      cleanupPlayer();
    };
  }, []);

  // Auto-scroll transcript
  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }, [messages]);

  const connectWebSocket = useCallback(() => {
    setStatus(STATES.CONNECTING);

    const ws = new WebSocket(`${WS_BASE}/ws/live`);

    ws.onopen = () => {
      console.log('[LiveScreen] WS connected, sending config lang=' + lang);
      ws.send(JSON.stringify({ type: 'config', language: lang }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log('[LiveScreen] WS msg:', msg.type, msg.text?.substring(0,50) || '');

        switch (msg.type) {
          case 'status':
            if (msg.status === 'ready') setStatus(STATES.READY);
            break;

          case 'user_transcript':
            setMessages(prev => [...prev, {
              id: `u_${Date.now()}`,
              role: 'user',
              text: msg.text,
            }]);
            break;

          case 'transcript':
            setMessages(prev => [...prev, {
              id: `a_${Date.now()}`,
              role: 'assistant',
              text: msg.text,
            }]);
            setStatus(STATES.RESPONDING);
            break;

          case 'audio_response':
            playResponseAudio(msg.data, msg.format);
            break;

          case 'turn_complete':
            // Don't set READY if audio is still playing
            if (status !== STATES.RESPONDING) setStatus(STATES.READY);
            break;

          case 'error':
            console.warn('Live WS error:', msg.message);
            setStatus(STATES.READY);
            break;
        }
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };

    ws.onerror = (e) => { console.warn('[LiveScreen] WS error', e.message); setStatus(STATES.IDLE); };
    ws.onclose = (e) => { console.log('[LiveScreen] WS closed', e.code, e.reason); setStatus(STATES.IDLE); };

    wsRef.current = ws;
  }, [lang]);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      // Stop any playing audio (interruption)
      cleanupPlayer();

      const perm = await requestRecordingPermissionsAsync();
      if (!perm.granted) {
        console.warn('Microphone permission denied');
        return;
      }

      recorder.record();
      console.log('[LiveScreen] Recording started');
      setStatus(STATES.RECORDING);
    } catch (e) {
      console.warn('Recording start error:', e.message);
    }
  }, [recorder]);

  // Stop recording and send audio
  const stopRecording = useCallback(async () => {
    try {
      await recorder.stop();
      const fileUri = recorder.uri;

      if (!fileUri) {
        setStatus(STATES.READY);
        return;
      }

      setStatus(STATES.PROCESSING);

      // Read recorded audio as base64
      const FileSystem = require('expo-file-system/legacy');
      console.log('[LiveScreen] Reading file:', fileUri);
      const base64Audio = await FileSystem.readAsStringAsync(fileUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      console.log('[LiveScreen] Audio base64 length:', base64Audio.length);

      // Send to backend via WebSocket
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        console.log('[LiveScreen] Sending audio via WS...');
        wsRef.current.send(JSON.stringify({
          type: 'audio',
          data: base64Audio,
          mime_type: 'audio/mp4',
        }));
      } else {
        setStatus(STATES.IDLE);
      }
    } catch (e) {
      console.warn('Recording stop error:', e.message);
      setStatus(STATES.READY);
    }
  }, [recorder]);

  // Play response audio
  const playResponseAudio = useCallback(async (base64Audio, format) => {
    try {
      const FileSystem = require('expo-file-system/legacy');
      const { createAudioPlayer } = require('expo-audio');

      const ext = format === 'wav' ? 'wav' : 'mp3';
      const fileUri = `${FileSystem.cacheDirectory}live_resp_${Date.now()}.${ext}`;
      await FileSystem.writeAsStringAsync(fileUri, base64Audio, {
        encoding: FileSystem.EncodingType.Base64,
      });

      cleanupPlayer();

      const player = createAudioPlayer(fileUri);
      playerRef.current = player;

      const sub = player.addListener('playbackStatusUpdate', (s) => {
        if (s.didJustFinish) {
          sub.remove();
          cleanupPlayer();
          setStatus(STATES.READY);
        }
      });

      player.play();
      setStatus(STATES.RESPONDING);
    } catch (e) {
      console.warn('Playback error:', e.message);
      setStatus(STATES.READY);
    }
  }, []);

  const cleanupPlayer = () => {
    if (playerRef.current) {
      try { playerRef.current.pause(); } catch {}
      try { playerRef.current.release(); } catch {}
      playerRef.current = null;
    }
  };

  // Mic press handlers
  const handleMicPressIn = () => {
    if (status === STATES.READY || status === STATES.RESPONDING) {
      startRecording();
    }
  };

  const handleMicPressOut = () => {
    if (status === STATES.RECORDING) {
      stopRecording();
    }
  };

  const handleReconnect = () => {
    if (status === STATES.IDLE) connectWebSocket();
  };

  // Status text
  const getStatusText = () => {
    const texts = {
      [STATES.IDLE]: lang === 'en' ? 'Disconnected. Tap to reconnect.' : 'Déconnecté. Appuie pour reconnecter.',
      [STATES.CONNECTING]: lang === 'en' ? 'Connecting...' : 'Connexion...',
      [STATES.READY]: lang === 'en' ? 'Hold to speak' : 'Maintiens pour parler',
      [STATES.RECORDING]: lang === 'en' ? 'Listening...' : "J'écoute...",
      [STATES.PROCESSING]: lang === 'en' ? "NAM SA' is thinking..." : "NAM SA' réfléchit...",
      [STATES.RESPONDING]: lang === 'en' ? "NAM SA' is speaking..." : "NAM SA' parle...",
    };
    return texts[status] || '';
  };

  const getStatusColor = () => {
    switch (status) {
      case STATES.RECORDING: return '#E53935';
      case STATES.PROCESSING: return Colors.secondary;
      case STATES.RESPONDING: return Colors.primary;
      case STATES.READY: return '#4CAF50';
      default: return Colors.textMuted;
    }
  };

  // Render message bubble
  const renderMessage = ({ item }) => {
    const isUser = item.role === 'user';
    return (
      <View style={[styles.msgRow, isUser ? styles.msgRowUser : styles.msgRowAssistant]}>
        {!isUser && (
          <View style={styles.avatar}>
            <Ionicons name="sunny" size={14} color={Colors.secondary} />
          </View>
        )}
        <View style={[styles.msgBubble, isUser ? styles.userBubble : styles.assistantBubble]}>
          <Text style={[styles.msgText, isUser ? styles.userText : styles.assistantText]}>
            {item.text}
          </Text>
        </View>
      </View>
    );
  };

  const micEnabled = status === STATES.READY || status === STATES.RESPONDING;

  // Ring animation interpolations
  const ringScale = ringAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [1, 2.2],
  });
  const ringOpacity = ringAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.4, 0],
  });

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle}>NAM SA' Live</Text>
          <View style={styles.statusRow}>
            <View style={[styles.statusDot, { backgroundColor: getStatusColor() }]} />
            <Text style={styles.headerSub}>
              {status === STATES.IDLE
                ? (lang === 'en' ? 'Offline' : 'Hors ligne')
                : (lang === 'en' ? 'Connected' : 'Connecté')}
            </Text>
          </View>
        </View>
        <TouchableOpacity
          onPress={() => navigation.navigate('Chat')}
          style={styles.chatBtn}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Ionicons name="chatbubble-ellipses-outline" size={20} color={Colors.textSecondary} />
        </TouchableOpacity>
      </View>

      {/* Transcript */}
      {messages.length > 0 ? (
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.transcriptList}
          showsVerticalScrollIndicator={false}
          style={styles.transcriptContainer}
        />
      ) : (
        <View style={styles.emptyTranscript}>
          <Ionicons name="mic-circle-outline" size={72} color={Colors.sand} />
          <Text style={styles.emptyTitle}>
            {lang === 'en' ? 'Voice Conversation' : 'Conversation Vocale'}
          </Text>
          <Text style={styles.emptyText}>
            {lang === 'en'
              ? "Hold the mic button and speak naturally.\nNAM SA' will respond with voice."
              : "Maintiens le micro et parle naturellement.\nNAM SA' te répondra par la voix."}
          </Text>
        </View>
      )}

      {/* Mic Area */}
      <View style={[styles.micArea, { paddingBottom: insets.bottom + Spacing.md }]}>
        <Text style={[styles.statusText, { color: getStatusColor() }]}>
          {getStatusText()}
        </Text>

        <View style={styles.micContainer}>
          {/* Expanding ring animation for recording */}
          {status === STATES.RECORDING && (
            <Animated.View
              style={[
                styles.ring,
                {
                  transform: [{ scale: ringScale }],
                  opacity: ringOpacity,
                },
              ]}
            />
          )}

          <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
            <TouchableOpacity
              style={[
                styles.micBtn,
                status === STATES.RECORDING && styles.micBtnRecording,
                status === STATES.PROCESSING && styles.micBtnProcessing,
                !micEnabled && status !== STATES.RECORDING && styles.micBtnDisabled,
              ]}
              onPressIn={handleMicPressIn}
              onPressOut={handleMicPressOut}
              onPress={status === STATES.IDLE ? handleReconnect : undefined}
              activeOpacity={0.7}
              disabled={status === STATES.CONNECTING || status === STATES.PROCESSING}
            >
              <Ionicons
                name={
                  status === STATES.RECORDING ? 'mic' :
                  status === STATES.PROCESSING ? 'hourglass-outline' :
                  'mic-outline'
                }
                size={36}
                color={Colors.textOnPrimary}
              />
            </TouchableOpacity>
          </Animated.View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },

  // Header
  header: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm,
    backgroundColor: Colors.warmWhite,
    borderBottomWidth: 1, borderBottomColor: Colors.sand,
  },
  backBtn: {
    width: 38, height: 38, borderRadius: 19,
    backgroundColor: Colors.card, justifyContent: 'center', alignItems: 'center',
    ...Shadows.sm,
  },
  headerCenter: { flex: 1, alignItems: 'center' },
  headerTitle: { ...Typography.h3, color: Colors.primary },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  headerSub: { ...Typography.caption, color: Colors.textMuted },
  chatBtn: {
    width: 38, height: 38, borderRadius: 19,
    backgroundColor: Colors.card, justifyContent: 'center', alignItems: 'center',
    ...Shadows.sm,
  },

  // Transcript
  transcriptContainer: { flex: 1 },
  transcriptList: {
    paddingHorizontal: Spacing.md, paddingVertical: Spacing.md, flexGrow: 1,
  },
  emptyTranscript: {
    flex: 1, justifyContent: 'center', alignItems: 'center',
    paddingHorizontal: Spacing.xl,
  },
  emptyTitle: {
    ...Typography.h3, color: Colors.textPrimary, marginTop: Spacing.md,
  },
  emptyText: {
    ...Typography.body, color: Colors.textMuted,
    textAlign: 'center', marginTop: Spacing.sm, lineHeight: 22,
  },

  // Messages
  msgRow: { flexDirection: 'row', marginBottom: Spacing.sm, alignItems: 'flex-end' },
  msgRowUser: { justifyContent: 'flex-end' },
  msgRowAssistant: { justifyContent: 'flex-start' },
  avatar: {
    width: 26, height: 26, borderRadius: 13,
    backgroundColor: Colors.sand, justifyContent: 'center', alignItems: 'center',
    marginRight: Spacing.xs, marginBottom: 2,
  },
  msgBubble: {
    maxWidth: SCREEN_W * 0.75, paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm, borderRadius: BorderRadius.lg,
  },
  userBubble: { backgroundColor: Colors.primary, borderBottomRightRadius: 4 },
  assistantBubble: {
    backgroundColor: Colors.card, borderBottomLeftRadius: 4,
    borderWidth: 1, borderColor: Colors.sand,
  },
  msgText: { ...Typography.body, lineHeight: 22 },
  userText: { color: Colors.textOnPrimary },
  assistantText: { color: Colors.textPrimary },

  // Mic area
  micArea: {
    alignItems: 'center', paddingTop: Spacing.md,
    backgroundColor: Colors.warmWhite,
    borderTopWidth: 1, borderTopColor: Colors.sand,
  },
  statusText: { ...Typography.label, marginBottom: Spacing.md },
  micContainer: {
    width: 120, height: 120, justifyContent: 'center', alignItems: 'center',
  },
  ring: {
    position: 'absolute', width: 88, height: 88, borderRadius: 44,
    backgroundColor: '#E53935',
  },
  micBtn: {
    width: 88, height: 88, borderRadius: 44,
    backgroundColor: Colors.secondary, justifyContent: 'center', alignItems: 'center',
    ...Shadows.glow,
  },
  micBtnRecording: { backgroundColor: '#E53935' },
  micBtnProcessing: { backgroundColor: Colors.stone },
  micBtnDisabled: { backgroundColor: Colors.stone, opacity: 0.6 },
});
