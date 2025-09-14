// Type definitions for audio libraries

declare module "mic" {
  interface MicOptions {
    rate?: string;
    channels?: string;
    debug?: boolean;
    exitOnSilence?: number;
  }

  interface MicInstance {
    start(): void;
    stop(): void;
    getAudioStream(): NodeJS.ReadableStream;
  }

  function mic(options?: MicOptions): MicInstance;
  export = mic;
}

declare module "speaker" {
  interface SpeakerOptions {
    channels?: number;
    bitDepth?: number;
    sampleRate?: number;
  }

  class Speaker {
    constructor(options?: SpeakerOptions);
    write(chunk: Buffer): boolean;
    end(): void;
  }

  export = Speaker;
}
