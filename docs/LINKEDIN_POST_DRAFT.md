# LinkedIn Post Draft

I have been working on a visual scanner prototype for TCG cards.

The idea is simple to explain, but technically interesting: point a camera at a card, detect the card region, normalize the image, and search it by visual similarity against a local index.

The current pipeline uses two main models:

1. YOLO to locate the full card in an image or video frame.
2. SigLIP 2 to turn the normalized card image into a visual embedding.

The flow looks like this:

- import Riftbound metadata and official card images;
- preprocess every card into a stable `384 x 384` input;
- generate embeddings with SigLIP 2;
- store vectors and metadata in LanceDB;
- when a user image arrives, detect the card, crop it, normalize it, generate its embedding, and retrieve the nearest neighbors.

I chose LanceDB because, for this prototype, it gives me fast local vector search with a setup that is easy to reproduce. I did not need to start with a distributed vector database before validating accuracy, latency, and the scanner experience.

One important design decision was to separate two problems:

- detecting where the card is;
- identifying which card it is.

For detection, I trained a single-class YOLO model (`card`) on a universal TCG card dataset. I mixed sources from MTG, Pokemon, Grand Archive, and other TCGs so the detector could learn the generic shape of a trading card instead of overfitting to one specific artwork style or set. The current prototype still needs more real Riftbound photos in the training data, but it already works well enough to validate the experience.

On the UI side, I built a live camera mode: the browser sends lightweight frames to the backend, YOLO returns the bounding box, and the border is drawn over the video. When the detection remains stable, the app captures a frame and runs the full visual search pipeline.

The performance work was also interesting. Warm embedding inference is already in the tens of milliseconds on my environment, so the bottleneck was not only the visual model. Some of the biggest improvements came from:

- keeping models loaded;
- compressing only the live detection frames;
- avoiding blocking recognition on price lookups;
- separating visual search from pricing requests.

The prototype also integrates prices through PriceCharting as an initial provider. For a real product, the next steps would be user accounts, collections by TCG and set, price history, portfolio metrics, and progress tracking for collectors.

It is still an MVP, but it already shows something powerful: you can build a fast card scanner by combining detection, visual embeddings, and vector search without depending on OCR or fragile rules over printed text.

Next step: clean the public repository, document the architecture properly, and keep evolving the product layer privately.
