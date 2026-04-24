
#
# установка Ollama, поднятие сервера и загрузка Qwen2.5:0.5B.
#

set -euo pipefail



# Константы
readonly OLLAMA_INSTALL_URL="https://ollama.com/install.sh"
readonly MODEL_NAME="qwen2.5:0.5b"
readonly API_ROOT="http://127.0.0.1:11434"
readonly STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-4}"



log() {
  # Вывод логов
  printf '[nlp-project] %s\n' "$*"
}



ensure_ollama_binary() {
  # Скачивание ollama 
  if command -v ollama >/dev/null 2>&1; then
    log "Ollama найдена: $(ollama --version 2>&1 | head -n1)"
    return 0
  fi
  curl -fsSL "${OLLAMA_INSTALL_URL}" | sh
}



ensure_ollama_daemon() {
  # Поднятие ollama HTTP-сервера

  if curl -fsS "${API_ROOT}/api/tags" >/dev/null 2>&1; then
    log "HTTP API Ollama уже поднята на ${API_ROOT}"
    return 0
  fi

  log "Запуск ollama сервера"
  nohup ollama serve >>/tmp/ollama-lab2.log 2>&1 &
  sleep "${STARTUP_WAIT_SEC}"
  if ! curl -fsS "${API_ROOT}/api/tags" >/dev/null 2>&1; then
    log "Ошибка: сервер не поднялся за ${STARTUP_WAIT_SEC} секунд"
    exit 1
  fi
  log "Сервер поднят"
}

pull_model() {
  # Загрузка модели

  log "Загрузка модели ${MODEL_NAME}..."
  ollama pull "${MODEL_NAME}"
}


main() {
  ensure_ollama_binary
  ensure_ollama_daemon
  pull_model
}

main "$@"
