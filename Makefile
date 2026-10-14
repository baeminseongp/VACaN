.PHONY: test validate atc-fixture atc-analytics atc-planning atc-evaluation caviar-evaluation step7-fixture jrdb-evaluation mall-evaluation web-data web-build web-media web-serve

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

validate:
	PYTHONPATH=src python -m vacan.validation data/synthetic/minimal

atc-fixture:
	PYTHONPATH=src python -m vacan.datasets.atc.cli map tests/fixtures/atc/localization_grid.yaml /tmp/vacan-atc-map
	PYTHONPATH=src python -m vacan.datasets.atc.cli trajectory tests/fixtures/atc/atc-sample.csv /tmp/vacan-atc-trajectories.csv
	PYTHONPATH=src python -m vacan.datasets.atc.cli align tests/fixtures/atc/localization_grid.yaml /tmp/vacan-atc-trajectories.csv /tmp/vacan-atc-alignment.json

atc-analytics:
	PYTHONPATH=src python -m vacan.analytics.cli configs/analytics/atc-step3.json

atc-planning:
	PYTHONPATH=src python -m vacan.planning.cli configs/planning/atc-step4.json

atc-evaluation:
	PYTHONPATH=src python -m vacan.evaluation.cli configs/evaluation/atc-step8-foundation.json

caviar-evaluation:
	PYTHONPATH=src python -m vacan.datasets.caviar.cli configs/datasets/caviar.json

step7-fixture:
	PYTHONPATH=src python -m vacan.datasets.jrdb.cli configs/evaluation/step7-jrdb-fixture.json
	PYTHONPATH=src python -m vacan.datasets.mall.cli configs/evaluation/step7-mall-fixture.json

jrdb-evaluation:
	PYTHONPATH=src python -m vacan.datasets.jrdb.cli configs/datasets/jrdb-step7.json

mall-evaluation:
	PYTHONPATH=src python -m vacan.datasets.mall.cli configs/datasets/mall-step7.json

web-data:
	python3 apps/web/scripts/export_research_data.py

web-build: web-data
	python3 apps/web/scripts/build_site.py

web-media:
	python3 apps/web/scripts/prepare_local_media.py

web-serve: web-build web-media
	python3 apps/web/scripts/serve_local.py
