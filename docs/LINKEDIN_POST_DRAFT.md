# LinkedIn Post Draft

I have been working on a visual scanner prototype for TCG cards.

The goal is simple from the user's point of view: point a camera at a card and get a reliable identification quickly. Under the hood, the interesting part is that I did not want to depend on OCR or brittle text rules. Cards can be sleeved, tilted, partially lit, printed in different languages, or photographed with motion blur. So I built the first version around visual search.

The current pipeline uses two main models:

1. YOLO to locate the full card in an image or video frame.
2. SigLIP 2 to convert the normalized card image into a visual embedding.

The important concept here is the visual embedding.

Instead of asking the model to output a card name directly, the model maps an image into a high-dimensional numerical vector. Images that look semantically similar should land close to each other in that vector space. That means we can preprocess the official card catalog once, store one embedding per card, and later compare a user photo against that index using nearest-neighbor search.

In practice, the flow looks like this:

- import Riftbound metadata and official card images;
- preprocess every card into a stable `384 x 384` input;
- generate a SigLIP 2 embedding for each reference card;
- store vectors and metadata in LanceDB;
- when a user image arrives, detect the card, crop it, normalize it, generate its embedding, and retrieve the closest cards from the vector index.

I chose this architecture because it separates the scanner into two problems that can be improved independently:

- detecting where the card is;
- identifying which card it is.

That separation matters. YOLO does not need to know whether the card is Riftbound, Pokemon, One Piece, or MTG. It only needs to learn the generic shape and boundaries of a trading card. For that reason, I trained a single-class YOLO model (`card`) on a universal TCG card dataset, mixing sources from MTG, Pokemon, Grand Archive, and other TCGs. The purpose was to make the detector generalize to "cardness" instead of overfitting to one specific set or artwork style.

The identification step has a different job. Once the card region is isolated, the embedding model captures visual signals from the full artwork, layout, colors, and printed structure. Then LanceDB handles the vector search and returns the nearest candidates with metadata attached.

I chose LanceDB for this prototype because it gives fast local vector search, simple reproducibility, and a low operational burden. At this stage, the priority was not to design a distributed database architecture too early. The priority was to validate precision, latency, and the scanner experience with a system that can be rebuilt and audited easily.

On the UI side, I built a live camera mode: the browser sends lightweight frames to the backend, YOLO returns the bounding box, and the border is drawn over the video. When the detection remains stable, the app captures a frame and runs the full visual search pipeline.

The performance work was also interesting. Warm embedding inference is already in the tens of milliseconds on my environment, so the bottleneck was not only the embedding model. Some of the biggest improvements came from:

- keeping models loaded;
- compressing only the live detection frames;
- avoiding blocking recognition on price lookups;
- separating visual search from pricing requests;
- keeping the final recognition image at a quality level that preserves enough visual detail for matching.

The prototype also integrates prices through PriceCharting as an initial provider. For a real product, the next steps would be user accounts, collections by TCG and set, price history, portfolio metrics, and progress tracking for collectors.

It is still an MVP, and the dataset still needs more real-world Riftbound photos, but the architecture already proves the core idea: a fast card scanner can be built by combining object detection, visual embeddings, and vector search without relying on OCR as the primary recognition mechanism.

Next step: clean the public repository, document the architecture properly, and keep evolving the product layer privately.
