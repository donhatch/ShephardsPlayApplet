#JAVAROOT=/usr/java/jdk1.5.0
#JAVAC=${JAVAROOT}/bin/javac

#JAVAROOT=/usr/java/j2sdk1.4.2
#JAVAC=${JAVAROOT}/bin/javac

#JAVAROOT=/usr/java/j2sdk1.4.2
#JAVAC=jikes +P -source 1.4 -classpath ${JAVAROOT}/jre/lib/rt.jar

#JAVAROOT=/opt/blackdown-jdk-1.4.2.03
#JAVAC=${JAVAROOT}/bin/javac

#JAVAC=javac1.2
#JAVAROOT=c:/jdk1.2.2

JAVAC=javac1.3
JAVAROOT=c:/jdk1.3.1_20



JARFILE = ShephardsPlayApplet.jar
CLASSES = \
        GraphicsAntiAliasingSetter.class \
        MyGraphics.class \
        Misc.class \
        ShephardsPlayApplet.class \
        ${NULL}
JAR_DEPENDS_ON = ${CLASSES}      macros.h Makefile javacpp javarenumber
JAR_CONTAINS = *.class *.prejava macros.h Makefile javacpp javarenumber

# XXX ARGH! why doesn't it work using -classpath .:./donhatchsw.jar ???
# XXX doing this instead for now, making com a symlink
# XXX to a dir that contains all the donhatchsw class files
JAR_CONTAINS += com

.PHONY: all
all: ${JARFILE}

${JARFILE}: Makefile META-INF/MANIFEST.MF ${CLASSES}
	${JAVAROOT}/bin/jar -cfm ${JARFILE} META-INF/MANIFEST.MF ${JAR_CONTAINS}

CPPFLAGS += -Wall -Werror
# The following seems to work but clutters up output and may be less portable
#CPPFLAGS += -pedantic -std=c99
# The following is too strict for me (requires #'s to be non-indented)
#CPPFLAGS += -Wtraditional

.SUFFIXES: .prejava .java .class
.prejava.class:
        # The following is the way to do it on linux I think
	#javacpp ${CPPFLAGS} ${JAVAC} -classpath ".:./donhatchsw.jar" $*.prejava
        # Need to do the following instead on cygwin... ?
	javacpp ${CPPFLAGS} ${JAVAC} $*.prejava

	javarenumber -v 0 $*.class
	# too slow... only do this in the production version
	# on second thought, try it, for now...
	# on third hand, it bombs with Couldn't open GraphicsAntiAliasingSetter$*.class because that one has no subclasses... argh.
	#@javarenumber -v -1 $*'$$'*.class

# Separate renumber target since renumbering all the subclass files
# on every recompile is slow :-(.  Usually I run "make renumber"
# after an exception, and then run the program again so I will get
# a stack trace with line numbers from the .prejava files instead of
# the .java files.

${JARFILE}.is_renumbered: $(JAR_DEPENDS_ON)
	javarenumber -v -1 *.class
	${JAVAROOT}/bin/jar -cfm $(JARFILE).is_renumbered META-INF/MANIFEST.MF ${JAR_CONTAINS}
	touch $@
.PHONY: renumber
renumber: $(JARFILE).is_renumbered


MyGraphics.class: macros.h Makefile donhatchsw.jar
GraphicsAntiAliasingSetter.class: macros.h Makefile donhatchsw.jar
ShephardsPlayApplet.class: macros.h Makefile donhatchsw.jar

SENDFILES = index.php $(JARFILE)
.PHONY: send
send: renumber
	sh -c "scp $(SENDFILES) hatch@plunk.org:public_html/ShephardsPlayApplet/."

.PHONY: clean
clean:
	rm -f core ${JARFILE} *.class *.java.lines *.java

