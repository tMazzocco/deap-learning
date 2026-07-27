import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="auto" />
      <Stack screenOptions={{ headerStyle: { backgroundColor: '#111' }, headerTintColor: '#fff' }}>
        <Stack.Screen name="index" options={{ title: 'Photo Classification PoC' }} />
        <Stack.Screen name="sample" options={{ title: 'Sample (upload)' }} />
        <Stack.Screen name="analyse" options={{ title: 'Analyse (live)' }} />
        <Stack.Screen name="settings" options={{ title: 'Settings' }} />
      </Stack>
    </>
  );
}
