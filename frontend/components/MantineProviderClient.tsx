"use client";
import { MantineProvider } from "@mantine/core";

export default function MantineProviderClient({ children }: { children: React.ReactNode }) {
  return (
    <MantineProvider>
      {children}
    </MantineProvider>
  );
}
