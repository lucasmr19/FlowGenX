La clave es separar dos conceptos distintos:

Cómo se segmenta el tráfico (flows vs ventanas de paquetes vs session ...)

Cómo se representa cada segmento (nPrint, GASF, tokens, etc.)

En tu código actual esos conceptos están mezclados, porque tu pipeline siempre termina en Flow.

PCAP
│
▼
PCAPReader
│
▼
PacketParser
│
▼
ParsedPacket
│
▼
Aggregator
├─ FlowAggregator
└─ PacketWindowAggregator
│
▼
TrafficSample
├─ Flow
└─ PacketWindow
│
▼
TrafficRepresentation
├─ SequentialRepresentation
├─ NprintRepresentation
└─ GASFRepresentation
│
▼
Tensor
│
▼
GenerativeModel

### Arquitectura data_utils

data/preprocessing.py — Pipeline completo PCAP → Flujos con 4 clases encadenadas: PCAPReader (streaming o en memoria con Scapy), PacketParser (extrae campos por capa con IAT), FlowAggregator (agrupación bidireccional por 5-tupla canónica con timeout) y FeatureNormalizer (min-max / z-score con fit separado para evitar data leakage). La clase PCAPPipeline los orquesta en un solo .process("file.pcap").
data/loaders.py — TrafficDataModule gestiona los splits train/val/test reproducibles y ajusta la representación solo sobre train antes de hacer encode. Los DataLoaders son estándar PyTorch. Hay un factory helper build_datamodule_from_pcap() que hace todo en una línea.
representations/base.py — Contrato abstracto con fit(), encode(), decode(), save() / load() y las propiedades invertibility e representation_type que son clave para la comparativa del TFG.
representations/sequential/tokenizer.py — FlatTokenizer (tokenización campo:valor con discretización percentil) y ProtocolAwareTokenizer (añade separadores de capa <L3>/<L4>, estados TCP atómicos y jerarquía protocolaria, inspirado en NetGPT).
representations/vision/representations.py — GASFRepresentation (NON_INVERTIBLE, arccos + gramian matrix sobre series temporales de IAT/tamaño) y NprintRepresentation (INVERTIBLE, serialización campo-a-bits con decode() funcional).
representations/**init**.py incluye un REGISTRY + get_representation("nprint") para que experiments/registry.py pueda instanciar representaciones por nombre desde configs YAML.
Siguiente módulo natural sería generative_models/ (Transformer autoregresivo para secuencial, UNet de difusión para visual) o evaluation/ si prefieres tener primero las métricas listas.

### Arquitectura generative_models

base.py — El contrato define train_step() devolviendo siempre {"loss": ..., ...extras}. Esto es lo que permite que el Trainer sea completamente agnóstico al modelo. Los modelos con múltiples optimizadores (GAN) sobreescriben configure_optimizers() devolviendo un dict en lugar de uno solo.
transformer/model.py — GPT-style con nn.TransformerDecoder en modo decoder-only: memory=tgt (sin cross-attention). Usa Pre-LN (norm_first=True) que es más estable que Post-LN en entrenamiento largo. Weight tying entre token_emb y lm_head reduce parámetros. La generación soporta top-k, nucleus sampling (top-p) y temperatura configurables.
diffusion/unet.py — UNet 2D con SinusoidalTimeEmbedding (inyectado por suma en cada ResidualBlock), GroupNorm + SiLU (más estables que BatchNorm + ReLU en difusión), y SelfAttentionBlock solo en los niveles bajos de resolución. El ConditioningEmbedding se suma al time embedding — esto replica el condicionamiento de NetDiffusion.
diffusion/ddpm.py — Implementa tanto el schedule coseno (Nichol & Dhariwal, 2021, más suave para imágenes de tráfico) como lineal. Muestreo DDPM estándar y DDIM acelerado (50 pasos por defecto en lugar de 1000). El DDIM con eta=0 es determinístico, útil para reproducibilidad en evaluación.
gan/model.py — WGAN-GP: el gradient penalty se calcula sobre embeddings interpolados (espacio continuo) porque las secuencias discretas no son directamente diferenciables. El generador usa Gumbel-Softmax durante entrenamiento para mantener diferenciabilidad.
Una cosa a tener en cuenta para los tests: el train_step() de la GAN espera que el Trainer asigne self.\_opt_discriminator y self.\_opt_generator antes de llamarlo — esto está documentado en el docstring y el Trainer del siguiente módulo lo gestionará automáticamente.

### Arquitectura evalutation

base.py define EvaluationResult → EvaluationReport → BaseEvaluator. Cualquier métrica nueva que añadas en el futuro solo requiere subclasear BaseEvaluator e implementar evaluate().
statistical.py (StatisticalEvaluator) opera siempre sobre tensores aplanados a (N, F), lo que lo hace agnóstico a la representación: funciona igual con [N, 64] (sequential), [N, 2, 16, 16] (GASF) o [N, 10, 181] (nprint). Calcula EMD y JS por feature y los agrega como media, más la distancia de Frobenius entre matrices de correlación de Pearson (captura si el modelo preserva dependencias entre features).
structural.py (StructuralEvaluator) es consciente de la representación y aplica reglas distintas según el tipo. Para nprint hay tres métricas: non_binary_field_rate (cuántos campos no son 0/1 antes de umbralizar), valid_sample_rate, y binarization_confidence (cuántos valores caen con claridad cerca de 0 o 1, no en la zona ambigua 0.4-0.6). Esta última es especialmente útil para evaluar si el DDPM ha aprendido la naturaleza binaria del espacio nprint.
downstream.py (TSTREvaluator) implementa TSTR completo con baseline TRTR opcional. El clasificador es un MLPClassifier de sklearn con PCA previo si hay más de 256 features, evitando que la dimensionalidad de nprint (10×181=1810) domine el tiempo de evaluación. El accuracy_gap = trtr - tstr es la métrica clave para el TFG: gap ≈ 0 significa que el sintético es tan útil como el real para tareas downstream.
suite.py (EvaluationSuite) orquesta los tres evaluadores y devuelve un SuiteResult con .summary() (dict plano), .get_metric() (búsqueda por nombre), y .to_dataframe() (pandas) para generar tablas comparativas directamente para el capítulo de resultados.
Una cosa que tendrás que ajustar: StructuralEvaluator(representation_type="sequential", vocab_size=rep.vocab_size) asume que FlatTokenizer expone el atributo vocab_size. Si no lo tiene, es un cambio de una línea en el tokenizador.
