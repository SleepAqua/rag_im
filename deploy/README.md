# Deploying the scheduled run

These units assume:

- the checkout lives at `/opt/rag-im`, owned by a `ragim` service user
- its virtualenv is `/opt/rag-im/.venv`
- state goes to `/var/lib/rag-im` (writable by `ragim`)

```bash
sudo useradd --system --home /var/lib/rag-im --shell /usr/sbin/nologin ragim
sudo install -d -o ragim -g ragim /var/lib/rag-im
sudo install -m 0644 deploy/rag-im.service /etc/systemd/system/
sudo install -m 0644 deploy/rag-im.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-im.timer
```

Check it:

```bash
systemctl list-timers rag-im.timer
journalctl -u rag-im.service -n 50
sudo -u ragim /opt/rag-im/.venv/bin/rag-im --data-dir /var/lib/rag-im status --metrics
```

`rag-im run` exits `0` when every source succeeded, `1` when at least one
source failed (the others still run), `3` when a previous run still holds the
lock, and `2` for a bad `sources.yaml`. The unit treats `3` as success.
