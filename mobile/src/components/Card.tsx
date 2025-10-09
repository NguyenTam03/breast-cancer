/**
 * Card Component with dark theme design
 */

import React from 'react';
import {
  View,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { colors } from '../theme/colors';

interface CardProps {
  children: React.ReactNode;
  style?: ViewStyle;
  padding?: 'none' | 'small' | 'medium' | 'large';
  shadow?: boolean;
}

export default function Card({
  children,
  style,
  padding = 'medium',
  shadow = true,
}: CardProps) {
  return (
    <View style={[
      styles.card,
      shadow && styles.shadow,
      styles[padding],
      style
    ]}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.background.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  shadow: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 6,
    },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 12,
  },
  none: {
    padding: 0,
  },
  small: {
    padding: 16,
  },
  medium: {
    padding: 24,
  },
  large: {
    padding: 32,
  },
});
