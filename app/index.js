import { Link } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';

export default function Home() {
  return (
    <View style={styles.container}>
      <Text style={styles.subtitle}>Pick a mode</Text>

      <Link href="/sample" style={[styles.button, styles.sample]}>
        <Text style={styles.buttonText}>📤  Sample</Text>
      </Link>
      <Text style={styles.hint}>Auto-capture and upload photos with an optional label.</Text>

      <Link href="/analyse" style={[styles.button, styles.analyse]}>
        <Text style={styles.buttonText}>🧠  Analyse</Text>
      </Link>
      <Text style={styles.hint}>Auto-capture and get a live label back from the model.</Text>

      <Link href="/settings" style={[styles.button, styles.settings]}>
        <Text style={styles.buttonText}>⚙️  Settings</Text>
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 8, justifyContent: 'center', backgroundColor: '#fff' },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 12, textAlign: 'center' },
  button: {
    paddingVertical: 18,
    paddingHorizontal: 20,
    borderRadius: 14,
    textAlign: 'center',
    overflow: 'hidden',
    marginTop: 8,
  },
  buttonText: { color: '#fff', fontSize: 20, fontWeight: '600' },
  sample: { backgroundColor: '#2563eb' },
  analyse: { backgroundColor: '#7c3aed' },
  settings: { backgroundColor: '#374151', marginTop: 24 },
  hint: { color: '#888', fontSize: 13, textAlign: 'center', marginBottom: 8 },
});
