USE atc;

create table if not exists clientes (
    id int not null auto_increment,
    nombre varchar(255) not null,
    email varchar(100) not null,
    primary key (id)
);

create table if not exists impagos (
    id int not null auto_increment,
    client_id int not null,
    fecha_impago date not null,
    primary key (id),
    foreign key (client_id) references clientes(id)
);

create table if not exists emails (
    id int not null auto_increment,
    fecha_envio datetime not null,
    client_id int not null,
    email varchar(4000) not null,
    primary key (id),
    foreign key (client_id) references clientes(id)
);

CREATE TABLE IF NOT EXISTS categorias (
    id INT NOT NULL AUTO_INCREMENT,
    nombre VARCHAR(255) NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS emails_categorias (
    email_id INT NOT NULL,
    categoria_id INT NOT NULL,
    PRIMARY KEY (email_id, categoria_id),
    FOREIGN KEY (email_id) REFERENCES emails(id),
    FOREIGN KEY (categoria_id) REFERENCES categorias(id)
);


INSERT INTO clientes (nombre, email)
VALUES ('Juan', 'juan@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Pedro', 'pedro@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Maria', 'maria@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Jose', 'jose@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Pepe', 'pepe@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Pablo', 'pablo@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Hugo', 'hugo@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Arnau', 'arnau@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Berta', 'berta@test.com');
INSERT INTO clientes (nombre, email)
VALUES ('Juana', 'juana@test.com');


INSERT INTO impagos (client_id, fecha_impago)
VALUES (1, '2021-01-01');
INSERT INTO impagos (client_id, fecha_impago)
VALUES (1, '2021-02-01');
INSERT INTO impagos (client_id, fecha_impago)
VALUES (8, '2021-01-01');
INSERT INTO impagos (client_id, fecha_impago)
VALUES (8, '2021-02-01');
INSERT INTO impagos (client_id, fecha_impago)
VALUES (8, '2021-03-01');

INSERT INTO categorias (nombre)
VALUES ("Facturas");
INSERT INTO categorias (nombre)
VALUES ("Problemas tecnicos");
INSERT INTO categorias (nombre)
VALUES ("Energia");
INSERT INTO categorias (nombre)
VALUES ("Gas");
INSERT INTO categorias (nombre)
VALUES ("Contrato");

INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-06-12 06:23:15', 1, 'Hola
Soy cliente pero no puedo acceder a mi factura on line y tampoco me puedo registrar como cliente porque siempre me  sale una ventana que me dice "error del servidor"
¿Me pueden decir si hay alguna forma de consultar mis facturas?
Podrían ponerlo un poco más fácil...
Atentamente
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-08-12 19:15:39', 2, 'Hola, buenos días. Necesito que alguien me ayude para poder ver mi factura. No me deja crear correo. Gracias. Cuando puedan, ayúdenme.
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-05-11 06:26:11', 3, 'Cada cuanto facturas el gas. Llevo con vosotros desde el 10 de febrero y no me a llegado ninguna factura.  
Enviado desde
Outlook para Android
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 4);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-04-15 19:13:26', 4, 'Buenos días,
No consigo ver la siguiente factura. Por favor enviarlo adjunto, gracias
Factura
Fecha
Cups
Importe
Ver
F-23-XXXXXXX
21/03/2023
ES00000054053403
490,90 €
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-04-07 19:25:24', 5, '
Hola buenas tardes,
seguimos desde que realizaron cambios en su web,
sin poder conectarnos y por tanto sin obtener nuestras facturas ni
tampoco adjuntar las lecturas mensuales.
 
Hemos solicitado en repetidas ocasiones y por diferentes medios que solventen el problema y todavía no tenemos
acceso y no hemos recibido ninguna contestación al respecto.
 
De nuevo nos vemos obligados a enviarles nuestras lecturas de GAS y LUZ por correo electrónico y solicitamos de nuevo
las dos facturas (en pdf) que ya han sido pagadas por nuestra entidad bancaria y que no tenemos.
 
FACTURAS : GAS por importe de 11,82 € y LUZ por importe de 153,49 €.
 
Así mismo adjuntamos nuestras lecturas tanto de gas como de luz, para la confección de nuestra factura de marzo.
 
Saludos
 
Admin
 
 
 
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2), (LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 4);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-02-18 19:29:31', 6, '
Porque motivo no puedo visualizar mi contrato después de firmar.Soy el firmante.Antonio Mel Lema,con DNI 12345678Z .Melo pueden enviar por correo electrónico.Gracias
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 2), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-05-12 20:01:26', 7, 'Me ha entrado la factura, ou sea, me ha entrado la factura, no, me ha llegado el cargo al banco de la luz y sin embargo no tengo las facturas. A ver si me las pueden mandar por correo, por favor. Vale, gracias a dios.
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2), (LAST_INSERT_ID(), 3);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-05-08 08:54:34', 8,
        'Hola, buenos días. Mira que me habéis cobrado las facturas de la luz y el gas, pero no me habéis mandado la factura. ¿Vale? Gracias.');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 4);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-05-23 19:11:41', 9, '

Buenos dias , nos ponemos en contacto con ustedes para solicitar documentación necesaria para la central : 

TITULAR : XXXX PRUEBA
NIF B123456789

REF CONTRATO :XXXXXXXX


Contrato de electricidad vigente, Facturas de energía de los últimos 12 meses, Curvas de consumo del último año
Quedamos a la espera , saludos!
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-08-08 20:42:46', 2, '
Buenos días, quería saber si tengo con vosotros contratado el gas.
Gracias 
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 4), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-01-25 19:57:15', 2, '
no he recibido el contrato 
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 2), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-08-12 08:52:39', 1, '
Buenos días
He borrado sin querer mis contratos de luz y gas..
Podrían volver a mandármelos?
Luis XXX
454135456P

Muchas gracias y perdona las molestias

Enviado desde mi iPhone
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 4), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-11-09 20:40:38', 3, '
Hola buenas tarde necesito que me envíe el contrato  de electricidad y gas nunca me lo han enviado 
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 2), (LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 4), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-04-18 07:11:29', 4, '
Hola, yo llamaba porque quiero darme de baja del servicio de gas. Entonces parece que he llegado un poco tarde, por eso es las seis y media. El teléfono mío es 6666666666. Si puede mañana me llamo. Gracias.

');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 4), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-01-05 06:49:43', 5, '
Señores:
Adjunto la documentación requerida para el cambio de titularidad del contrato de suministro de electricidad ( nº ZXXXXXX), actualmente a nombre del Sr. Salvador , relativo al local sito en Barcelona, c/.Prueba.

Atentamente,
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 5);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-03-15 08:38:44', 6, '
Buenas tardes,
Necesito que me envieis las facturas de gas y luz del mes de abril por favor, me tengo que ir del piso en el que vivo y el propietario las necesita para hacer cambio de titularidad .
Gracias 
');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 3), (LAST_INSERT_ID(), 4);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-08-18 20:57:43', 7, 'Buenos días. Solicitamos nos envíen la factura XXXXXX de importe XX€ emitida a nuestra empresa Panaderías XXXXX S.L.');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-08-13 06:56:49', 8,
        'Buenos días, Necesitamos que nos envíen la factura del mes de septiembre del siguiente cliente: XXXXXX');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-09-15 06:19:16', 9,
        'Buenos días. Soy Pepito Grillo. Llevo dos días intentando conseguir mis facturas por mail o por teléfono y no me es posible. Por favor se pueden poner en contacto conmigo?? Gracias');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-08-08 19:22:27', 8, 'Buenos días. Quisiera que me enviarán también las siguientes facturas:

- recibida 18/07/2023, importe XX€
- recibida 18/07/2023, importe YY€
- recibida 18/08/2023, importe ZZ€');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-07-13 18:21:44', 1, 'Necesito hablar con un operador que es urgente, necesito ver las facturas que tengo de luz.');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 3);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-03-13 07:24:42', 2, 'Buenas tardes. Estoy pidiendo a la factura que me habéis cargado el 25 de septiembre con importe de XX. Mi empresa es Panadería González S.L. Por favor, enviarme la factura, que me urge.');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1);


INSERT INTO emails (fecha_envio, client_id, email)
VALUES ('2022-06-17 06:07:44', 3, 'Hola, llamo porque he llamado antes, pero me ha atendido una operadora, pero se me ha cortado. Para que me digan la factura de este mes, ¿cuánto es? ¿Cuánto es? Mi número de teléfono lo tienen ustedes ahí. Bueno, tienen el móvil y el de casa, así que si pueden me llaman, gracias, para saber lo que tengo que pagar este mes, ¿vale? Gracias.');

INSERT INTO emails_categorias (email_id, categoria_id) VALUES
(LAST_INSERT_ID(), 1), (LAST_INSERT_ID(), 2);

