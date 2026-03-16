/**
 * DictionaryScreen — NAM SA'
 * Google Translate-style interface: 2 text boxes, language swap, TTS.
 */

import React, { useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, TextInput,
  ActivityIndicator, Platform, KeyboardAvoidingView,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useLanguage } from '../context/LanguageContext';
import { translate as apiTranslate, fetchTTS } from '../services/api';
import { useAudioPlayer } from 'expo-audio';
import * as FileSystem from 'expo-file-system';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '../theme';

const LANGUAGES = [
  { code: 'fr', label: 'Français' },
  { code: 'en', label: 'English' },
  { code: 'bbj', label: "Ghomala'" },
];

export default function DictionaryScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const { t } = useLanguage();

  const [sourceLang, setSourceLang] = useState('fr');
  const [targetLang, setTargetLang] = useState('bbj');
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [playingId, setPlayingId] = useState(null);

  const player = useAudioPlayer();

  const handleSwap = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setSourceText(translatedText);
    setTranslatedText(sourceText);
  };

  const cycleLang = (current, other) => {
    const codes = LANGUAGES.map((l) => l.code);
    let idx = codes.indexOf(current);
    let next;
    do {
      idx = (idx + 1) % codes.length;
      next = codes[idx];
    } while (next === other);
    return next;
  };

  const handleTranslate = useCallback(async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    try {
      const result = await apiTranslate(sourceText.trim(), sourceLang, targetLang);
      setTranslatedText(result.translation || '');
    } catch {
      setTranslatedText(t('error'));
    }
    setLoading(false);
  }, [sourceText, sourceLang, targetLang, t]);

  const playAudio = async (text, lang, id) => {
    if (playingId === id) return;
    setPlayingId(id);
    try {
      const result = await fetchTTS(text, lang);
      if (result.audio) {
        const ext = (result.mime_type || '').includes('wav') ? 'wav' : 'mp3';
        const fileUri = `${FileSystem.cacheDirectory}tts_dict_${Date.now()}.${ext}`;
        await FileSystem.writeAsStringAsync(fileUri, result.audio, {
          encoding: FileSystem.EncodingType.Base64,
        });
        player.replace({ uri: fileUri });
        player.play();
      }
    } catch (e) {
      console.warn('TTS failed:', e.message);
    }
    setPlayingId(null);
  };

  const getLangLabel = (code) => LANGUAGES.find((l) => l.code === code)?.label || code;

  return (
    <KeyboardAvoidingView
      style={[styles.container, { paddingTop: insets.top }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={22} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('translate')}</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Language Selector Row */}
      <View style={styles.langRow}>
        <TouchableOpacity
          style={styles.langPill}
          onPress={() => setSourceLang(cycleLang(sourceLang, targetLang))}
        >
          <Text style={styles.langPillText}>{getLangLabel(sourceLang)}</Text>
          <Ionicons name="chevron-down" size={14} color={Colors.textSecondary} />
        </TouchableOpacity>

        <TouchableOpacity onPress={handleSwap} style={styles.swapBtn}>
          <Ionicons name="swap-horizontal" size={26} color={Colors.secondary} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.langPill}
          onPress={() => setTargetLang(cycleLang(targetLang, sourceLang))}
        >
          <Text style={styles.langPillText}>{getLangLabel(targetLang)}</Text>
          <Ionicons name="chevron-down" size={14} color={Colors.textSecondary} />
        </TouchableOpacity>
      </View>

      {/* Source Text Box */}
      <View style={styles.textBox}>
        <TextInput
          style={styles.textInput}
          value={sourceText}
          onChangeText={(txt) => {
            setSourceText(txt);
            if (!txt.trim()) setTranslatedText('');
          }}
          placeholder={t('typeMessage')}
          placeholderTextColor={Colors.textMuted}
          multiline
          maxLength={1000}
          returnKeyType="done"
          blurOnSubmit
          onSubmitEditing={handleTranslate}
        />
        <View style={styles.textActions}>
          {sourceText.length > 0 && (
            <>
              <TouchableOpacity
                onPress={() => playAudio(sourceText, sourceLang, 'src')}
                disabled={playingId === 'src'}
              >
                <Ionicons
                  name={playingId === 'src' ? 'volume-high' : 'volume-medium-outline'}
                  size={22}
                  color={Colors.accent}
                />
              </TouchableOpacity>
              <TouchableOpacity
                onPress={() => {
                  setSourceText('');
                  setTranslatedText('');
                }}
              >
                <Ionicons name="close-circle" size={22} color={Colors.textMuted} />
              </TouchableOpacity>
            </>
          )}
        </View>
      </View>

      {/* Translate Button */}
      <TouchableOpacity
        style={[styles.translateBtn, !sourceText.trim() && styles.translateBtnDisabled]}
        onPress={handleTranslate}
        disabled={!sourceText.trim() || loading}
      >
        {loading ? (
          <ActivityIndicator color={Colors.textOnPrimary} />
        ) : (
          <>
            <Ionicons name="language" size={20} color={Colors.textOnPrimary} />
            <Text style={styles.translateBtnText}>{t('translateAction')}</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Translation Result Box */}
      <View style={[styles.textBox, styles.resultBox]}>
        <Text
          style={[
            styles.resultText,
            !translatedText && styles.placeholderResultText,
          ]}
          selectable
        >
          {translatedText || t('noTranslation')}
        </Text>
        {translatedText.length > 0 && (
          <View style={styles.textActions}>
            <TouchableOpacity
              onPress={() => playAudio(translatedText, targetLang, 'tgt')}
              disabled={playingId === 'tgt'}
            >
              <Ionicons
                name={playingId === 'tgt' ? 'volume-high' : 'volume-medium-outline'}
                size={24}
                color={Colors.primary}
              />
            </TouchableOpacity>
          </View>
        )}
      </View>

      {/* Bottom padding */}
      <View style={{ height: insets.bottom + Spacing.md }} />
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background, paddingHorizontal: Spacing.lg },
  // Header
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
  },
  backBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: Colors.card, justifyContent: 'center', alignItems: 'center',
    ...Shadows.sm,
  },
  headerTitle: { ...Typography.h2, color: Colors.primary },
  // Language row
  langRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: Spacing.md, marginVertical: Spacing.md,
  },
  langPill: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: Colors.card, paddingHorizontal: 16, paddingVertical: 10,
    borderRadius: BorderRadius.full, ...Shadows.sm,
  },
  langPillText: { ...Typography.label, color: Colors.textPrimary },
  swapBtn: {
    width: 44, height: 44, borderRadius: 22,
    backgroundColor: Colors.sand, justifyContent: 'center', alignItems: 'center',
  },
  // Text boxes
  textBox: {
    backgroundColor: Colors.card, borderRadius: BorderRadius.lg,
    padding: Spacing.md, minHeight: 120, borderWidth: 1,
    borderColor: Colors.sand, ...Shadows.sm,
  },
  textInput: {
    ...Typography.bodyLarge, color: Colors.textPrimary, flex: 1,
    minHeight: 80, textAlignVertical: 'top',
  },
  textActions: {
    flexDirection: 'row', justifyContent: 'flex-end', gap: Spacing.md, marginTop: Spacing.xs,
  },
  resultBox: { marginTop: Spacing.md, backgroundColor: Colors.warmWhite, flex: 1 },
  resultText: { ...Typography.bodyLarge, color: Colors.primary, lineHeight: 28 },
  placeholderResultText: { color: Colors.textMuted, fontStyle: 'italic' },
  // Translate button
  translateBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: Spacing.xs, backgroundColor: Colors.primary, borderRadius: BorderRadius.full,
    paddingVertical: 14, marginTop: Spacing.md, ...Shadows.md,
  },
  translateBtnDisabled: { backgroundColor: Colors.stone },
  translateBtnText: { ...Typography.label, color: Colors.textOnPrimary },
});
