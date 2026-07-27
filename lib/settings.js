import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = 'poc.settings.v1';

export const DEFAULT_SETTINGS = {
  // Base URL of the backend. Routes /sample and /analyse are appended.
  baseUrl: 'http://192.168.1.10:8000',
  // Seconds between each auto-capture.
  intervalSec: 5,
};

export async function loadSettings() {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export async function saveSettings(settings) {
  const merged = { ...DEFAULT_SETTINGS, ...settings };
  await AsyncStorage.setItem(KEY, JSON.stringify(merged));
  return merged;
}
