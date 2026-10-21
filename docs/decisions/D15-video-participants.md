# D15: quem aparece nos vídeos

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Juntas

## Contexto

Cada parte tem um vídeo público. O da Parte 1 apresenta problema, datasets, arquitetura, experimentos, resultados e a escolha do modelo (seção 35). O da Parte 2 mostra a aplicação funcionando: inicialização, cadastro, reconhecimento, uma **pessoa desconhecida**, uma situação de erro e a arquitetura (seção 42). As regras de privacidade valem também para o vídeo: imagens de terceiros só podem aparecer se forem compatíveis com o consentimento, e a seção 42 sugere usar o próprio aluno ou material não privado. O termo (Anexo A) trata a autorização para aparecer em vídeo separadamente da autorização para o desenvolvimento, e cada participante pode negar.

## Decisão

**Só Paola e Valentina aparecem nos vídeos.** Nenhuma participante do dataset privado aparece, mesmo que tenha autorizado no termo.

- **Vídeo da Parte 1:** sem fotos de participantes. O dataset privado aparece só em números agregados e na matriz de confusão com IDs anônimos (`P01`). Exemplos de imagens, se houver, vêm do LFW e apenas para ilustrar o pipeline.
- **Vídeo da Parte 2:**
  - **cadastro e reconhecimento:** uma de nós se cadastra pela câmera e depois é reconhecida;
  - **pessoa desconhecida:** a outra aparece **antes** de se cadastrar e recebe "pessoa desconhecida";
  - **situação de erro:** uma foto sem rosto ou com nós duas juntas (vários rostos);
  - o banco da demonstração tem só nós duas e é apagado depois da gravação, usando a própria exclusão da aplicação, o que também demonstra a seção 39.
- **Na tela:** nada do cofre, dos termos, da planilha ou de nomes de participantes. Terminal e explorador de arquivos só mostram código e dados públicos.

## Alternativas consideradas

- **Usar participantes que marcaram "AUTORIZO" no termo:** seria permitido, mas cria dependência de terceiros para gravar e para regravar, e um vídeo público com o rosto delas não pode ser "desfeito" se alguém retirar o consentimento depois.
- **Fotos de celebridades (do LFW ou da internet) como "desconhecida":** são imagens de terceiros, com direitos dos fotógrafos, e não mostram o fluxo real da câmera. Ter uma de nós como a desconhecida resolve o caso com a mesma câmera e a mesma iluminação.
- **Vídeo só com capturas de tela, sem rosto nenhum:** não demonstraria o reconhecimento pela câmera, que é o centro da Parte 2.

## Consequências

- Nenhuma participante precisa estar disponível nas gravações, e o vídeo não depende da autorização de vídeo do termo.
- Nós duas consentimos em aparecer por sermos as autoras e por gravarmos nós mesmas. O cadastro de demonstração é feito só para o vídeo e não entra nos experimentos.
- O roteiro dos vídeos (#32 e #51) segue esta decisão, e o checklist do Anexo B ("o vídeo respeita as autorizações concedidas") aponta para cá e para o [privacy.md](../privacy.md).
