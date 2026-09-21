# Always-on hosting options

Checked September 21, 2026. These are published compute prices, not a purchase or a guaranteed availability quote. Add applicable taxes, backups, domain and any extra disk/egress charges.

The service uses a single persistent-volume host with an HTTP process and one CPU inference worker. Docker Compose restarts crashed processes. Monitor `/healthz` for HTTP and `/readyz` for inference; keep the host awake, the Docker daemon enabled on boot, and platform autostop disabled. This provides always-on operation on one machine, not multi-node high availability or zero-downtime upgrades.

| Option | Published compute price | Fit / limitation |
| --- | --- | --- |
| Existing Linux server with spare 16 GiB RAM | Existing bill | Best if available; inspect spare CPU, RAM and disk before deploying alongside other services. |
| DigitalOcean Basic, 4 vCPU / 8 GiB | $48/month | Bounded low-volume pilot; lower container limit to 6 GiB to leave room for the OS. Full-book limits still unmeasured. |
| DigitalOcean Basic, 8 vCPU / 16 GiB | $96/month | More memory headroom; shared CPU means variable throughput. Default four-thread worker can use this host. |
| Hetzner CCX23, EU | $101.49/month excluding IPv4 and VAT | Dedicated CPU candidate for sustained inference; confirm current region capacity and final order quote. |
| Hetzner CX43 / CAX31, EU | Listed $18.49 / $24.99 monthly excluding IPv4 and VAT | Attractive budget candidates, but the cost-optimized tier currently shows unavailable. Do not assume we can provision them. |
| Render paid service with persistent disk | Price depends on selected CPU/RAM plan, plus disk | Less host administration; run API and worker in the same service because its disk is attached to one instance. Avoid free/sleeping plans. |

Recommendation: reuse an existing suitable host first. Otherwise choose the 16 GiB CPU VM if the budget permits; an 8 GiB pilot is an explicit lower-capacity alternative. Check Hetzner's low-cost stock at purchase time rather than delaying indefinitely for it. GPU hosting needs a separate performance and VRAM benchmark; it is not justified by the CPU excerpt pilot alone.

Sources:

- [DigitalOcean Droplet pricing](https://www.digitalocean.com/pricing/droplets)
- [Hetzner June 2026 price table](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)
- [Hetzner cost-optimized availability](https://www.hetzner.com/cloud/cost-optimized/)
- [Render compute plans](https://render.com/docs/compute-plans)
- [Render persistent disks](https://render.com/docs/disks)
- [Fly.io long-running tasks and autostop](https://fly.io/docs/blueprints/long-running-tasks/)

## Deployment decision still needed

No provider account, server target or spending ceiling has been selected in this task. Before provisioning, obtain that choice and access through the provider's normal login/secret mechanism. Do not paste tokens into chat. Then provision the host, deploy the reviewed image/commit with persistent storage and HTTPS, run the authenticated smoke test from outside the host, restart the service and verify cached job persistence. Record the actual endpoint and checks before calling the public API deployed.
