
# Uncomment one of the following pairs.

#JAVAROOT=/usr/java/jdk1.5.0
#JAVAC=${JAVAROOT}/bin/javac
#JAR=${JAVAROOT}/bin/jar

#JAVAROOT=/usr/java/j2sdk1.4.2
#JAVAC=${JAVAROOT}/bin/javac
#JAR=${JAVAROOT}/bin/jar

#JAVAROOT=/usr/java/j2sdk1.4.2
#JAVAC=jikes +P -source 1.4 -classpath ${JAVAROOT}/jre/lib/rt.jar
        # ARGH, doesn't work any more, conflicts with -classpath now used in individual rule
#JAR=${JAVAROOT}/bin/jar

#JAVAROOT=/opt/blackdown-jdk-1.4.2.03
#JAVAC=${JAVAROOT}/bin/javac
#JAR=${JAVAROOT}/bin/jar

# doesn't work for this project--- JFrame.EXIT_ON_CLOSE not defined
#JAVAC=javac1.2
#JAVAROOT=c:/jdk1.2.2
#JAR=${JAVAROOT}/bin/jar

#JAVAC=javac1.3
#JAVAROOT=c:/jdk1.3.1_20
#JAR=${JAVAROOT}/bin/jar

#JAVAC=javac1.6
#JAVAROOT="c:/Program Files (x86)/Java/jdk1.6.0_17"
#JAR=${JAVAROOT}/bin/jar

JAVAC=/usr/bin/javac
JAR=/usr/bin/jar


# On linux and cygwin, use uname -o; on darwin, use uname
uname := $(shell uname -o > /dev/null 2>&1 && uname -o || uname)
#dummy := $(warning uname = $(uname))
ifeq ($(uname),Cygwin)
    # on cygwin, apparently it's this
    CLASSPATHSEP = ;
else
    # on linux and darwin, it's this
    CLASSPATHSEP = :
endif



JARFILE = ShephardsPlayApplet.jar
CLASSES = \
        GraphicsAntiAliasingSetter.class \
        MyGraphics.class \
        MyGraphics3D.class \
        Misc.class \
        ExactTrig.class \
        GeomUtils.class \
        SymmetryUtils.class \
        SizeTrackingMergeFind.class \
        SymmetryUtils.class \
        Mesh.class \
        MeshUtils.class \
        Net.class \
        MyAlgorithmMaybe.class \
        Surface.class \
        ShephardsPlayApplet.class \
        PoincareDiskIsometry.class \
        PoincareHalfSpaceIdealCenter.class \
        ${NULL}
JAR_DEPENDS_ON = ${CLASSES}      macros.h Makefile javacpp javarenumber README.md
JAR_CONTAINS = *.class *.prejava macros.h Makefile javacpp javarenumber README.md

# If we want to be able to run it as java -jar ShephardsPlayApplet.jar, then need to do this:
# XXX ouch, but this makes "make renumber" fail
#JAR_CONTAINS += com

.PHONY: all default jar
default: Makefile ${JAR_DEPENDS_ON}
jar: ${JARFILE}
all: jar

${JARFILE}: Makefile META-INF/MANIFEST.MF ${JAR_DEPENDS_ON}
        # XXX argh, exits with status 0 even if missing something
	${JAR} -cfm ${JARFILE} META-INF/MANIFEST.MF ${JAR_CONTAINS}

CPPFLAGS += -Wall -Werror
# The following seems to work but clutters up output and may be less portable
#CPPFLAGS += -pedantic -std=c99
# The following is too strict for me (requires #'s to be non-indented)
#CPPFLAGS += -Wtraditional

CPPFLAGS+=-DOVERRIDE=@Override

.SUFFIXES: .prejava .java .class
.prejava.class:
	./javacpp ${CPPFLAGS} ${JAVAC} -deprecation -classpath ".$(CLASSPATHSEP)./donhatchsw.jar" $*.prejava
ifneq ($(uname),Cygwin)
	./javarenumber -v 0 $*.class
        # too slow... only do this in the production version
        # on second thought, try it, for now...
        # on third hand, it bombs with Couldn't open GraphicsAntiAliasingSetter$*.class because that one has no subclasses... argh.
        #@./javarenumber -v -1 $*'$$'*.class
endif

# Separate renumber target since renumbering all the subclass files
# on every recompile is slow :-(.  Usually I run "make renumber"
# after an exception, and then run the program again so I will get
# a stack trace with line numbers from the .prejava files instead of
# the .java files.

${JARFILE}.is_renumbered: $(JAR_DEPENDS_ON)
	./javarenumber -v -1 *.class
	${JAR} -cfm $(JARFILE).is_renumbered META-INF/MANIFEST.MF ${JAR_CONTAINS}
	touch $@
.PHONY: renumber
renumber: $(JARFILE).is_renumbered


MyGraphics.class: macros.h Makefile donhatchsw.jar
MyGraphics3D.class: macros.h Makefile donhatchsw.jar
GraphicsAntiAliasingSetter.class: macros.h Makefile donhatchsw.jar
ShephardsPlayApplet.class: macros.h Makefile donhatchsw.jar

SENDFILES = index.php $(JARFILE)
.PHONY: send
send: renumber
	sh -c "scp $(SENDFILES) hatch@plunk.org:public_html/ShephardsPlayApplet/."

.PHONY: clean
clean:
	rm -f core ${JARFILE} *.class *.java.lines *.java

