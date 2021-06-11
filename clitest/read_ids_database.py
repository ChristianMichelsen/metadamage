import logging

import pickledb


logger = logging.getLogger(__name__)


class DB:
    def __init__(self, cfg):
        self.path = cfg.file_read_ids_database
        self.db = pickledb.load(self.path, auto_dump=True)

    def load(self, tax_id):
        return self.db.lgetall(str(tax_id))[0]

    def save(self, d):
        logger.info(f"saving read ids to database: {self.path}")
        for tax_id, read_ids in d.items():
            self.db.lcreate(str(tax_id))
            self.db.ladd(str(tax_id), read_ids)
