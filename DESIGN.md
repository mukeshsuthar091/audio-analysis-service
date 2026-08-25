# Design write-up

The service uses FastAPI for a clear REST API and request validation. FFmpeg accepts common call formats and converts them to mono, 16 kHz audio. A direct WAV path avoids FFmpeg when the file is already normalized. Silero VAD finds usable speech and skips silence. I chose the audEERING six-layer Wav2Vec2 model because it produces age and gender outputs directly and is smaller than the twenty-four-layer version. Poor audio lowers confidence, and uncertain results become `unknown`.

With more time, I would test licensed logistics-call recordings with engines, road noise, warehouses, phone codecs, and varied speakers. I would measure accuracy and fairness across groups, tune quality limits, and calibrate confidence scores. I would compare smaller models, quantization, ONNX Runtime, and GPU inference. Speaker-change detection could be added after checking its speed, accuracy, and privacy cost.

To handle 1,000 calls, stateless APIs would sit behind a load balancer. Workers would preload the model on CPUs or GPUs. Requests would use bounded queues and backpressure, preventing unlimited memory use. The platform would add workers as queue depth or latency rises and remove them when demand falls. Metrics track errors, timeouts, queue age, readiness, and response time. This container does not support that load.
