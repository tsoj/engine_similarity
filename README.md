
# Chess engine search similarity

![alt text](similarity_graph_March-2025.jpeg)

## Setup

```bash
conda create -n engine_similarity python=3.12
conda activate engine_similarity
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install accelerate seaborn numpy scikit-learn matplotlib sentence_transformers opencv-python einops anthropic
```

# Run

```bash
python -u get_search_files.py --repos \
https://github.com/tsoj/Nalwald \
https://github.com/jhonnold/berserk \
https://github.com/Aryan1508/Bit-Genie \
https://github.com/jnlt3/blackmarlin \
https://github.com/lucasart/Demolito \
https://github.com/justNo4b/Drofa \
https://github.com/justNo4b/Equisetum \
https://github.com/AndyGrant/Ethereal \
https://github.com/fabianvdW/FabChess \
https://github.com/KierenP/Halogen \
https://github.com/vshcherbyna/igel \
https://github.com/Luecx/Koivisto \
https://github.com/jeffreyan11/laser-chess-engine \
https://github.com/Matthies/RubiChess \
https://github.com/connormcmonigle/seer-nnue \
https://github.com/mhouppin/stash-bot \
https://github.com/official-stockfish/Stockfish \
https://github.com/TerjeKir/weiss \
https://github.com/rosenthj/Winter \
https://github.com/amanjpro/zahak \
https://github.com/jw1912/akimbo \
https://github.com/PGG106/Alexandria \
https://github.com/Alex2262/AltairChessEngine \
https://github.com/dede1751/carp \
https://github.com/GediminasMasaitis/chess-dot-cpp \
https://github.com/lucametehau/CloverEngine \
https://github.com/spamdrew128/Galumph \
https://github.com/archishou/MidnightChessEngine \
https://github.com/Adam-Kulju/Patricia \
https://github.com/Ciekce/Polaris \
https://github.com/Ciekce/Stoat \
https://github.com/Ciekce/Stormphrax \
https://github.com/crippa1337/svart \
https://github.com/cosmobobak/viridithas \
https://github.com/spamdrew128/Wahoo \
https://github.com/Adam-Kulju/Willow \
https://github.com/aronpetko/Integral \
https://github.com/Yoshie2000/PlentyChess \
https://github.com/gab8192/Obsidian \
https://github.com/ArjunBasandrai/elixir-chess-engine \
https://github.com/xu-shawn/Serendipity \
https://github.com/yl25946/spaghet \
https://github.com/Quanticade/Quanticade \
https://github.com/Jochengehtab/Schoenemann \
https://github.com/mergener/illumina \
https://github.com/Vast342/Clarity \
https://github.com/analog-hors/Boychesser \
https://github.com/DeveloperPaul123/byte-knight \
https://github.com/folkertvanheusden/Dog \
https://github.com/JonathanHallstrom/pawnocchio \
https://github.com/analog-hors/tantabus \
https://github.com/dannyhammer/toad \
https://github.com/yukarichess/yukari \
https://github.com/87flowers/bannou \
https://github.com/MinusKelvin/ice4 \
https://github.com/zzzzz151/Starzix \
https://github.com/martinnovaak/motor \
https://github.com/pkrisz99/Renegade \
https://github.com/Mcthouacbb/Sirius \
https://github.com/liamt19/Horsie \
https://github.com/liamt19/Lizard \
https://github.com/liamt19/Peeper \
https://github.com/lynx-chess/Lynx \
https://github.com/rn5f107s2/Molybdenum \
https://github.com/jgilchrist/tcheran \
https://github.com/kelseyde/calvin-chess-engine \
https://github.com/nocturn9x/heimdall \
https://github.com/Quinniboi10/Prelude \
https://github.com/ProgramciDusunur/Potential \
https://github.com/ksw0518/Turbulence_v4 \
https://github.com/Altanis/sacre_dieu \
https://github.com/toanth/motors \
https://github.com/Mathmagician8191/Liberty-Chess \
https://github.com/kz04px/Baislicka \
https://github.com/kz04px/autaxx \
https://github.com/Ciekce/sanctaphraxx \
https://github.com/zzzzz151/Zataxx \
https://github.com/Algorhythm-sxv/Cheers \
https://github.com/raklaptudirm/mess \
https://github.com/spamdrew128/Apotheosis \
https://github.com/MinusKelvin/frozenight \
https://github.com/lithander/Leorik \
https://github.com/Sazgr/peacekeeper \
https://github.com/rafid-dev/rice \
https://github.com/Disservin/Smallbrain \
https://github.com/JoAnnP38/PedanticRF \
https://github.com/SnowballSH/Avalanche \
https://github.com/Witek902/Caissa \
https://github.com/brunocodutra/cinder \
https://github.com/kz04px/4ku \
https://github.com/TheRealGioviok/Perseus-Engine \
https://github.com/phenri/glaurung \
https://github.com/Warpten/Fruit-2.1 \
--out-dir ./raw_search_files \
--temp-dir ./engine_similarity_temp

python -u translate_files_to_cpp.py ./raw_search_files ./translated_search_files

python -u clang_format_files.py ./translated_search_files ./formatted_search_files

python -u compute_embeddings.py ./formatted_search_files ./embeddings.npz

python -u visualize_embeddings.py ./embeddings.npz --output-graph ./similarity_graph.png
```
